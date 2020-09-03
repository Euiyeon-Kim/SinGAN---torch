import os
import math

import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model.generator import Generator
from model.discriminator import Discriminator
from utils.loss import calcul_gp
from utils.layers import weights_init, reset_grads
from utils.image import read_img, resize_img, torch2np
from utils.utils import creat_reals_pyramid, generate_dir2save, generate_noise


class SinGAN:
    def __init__(self, config):
        self.config = config
        self.Gs = []
        self.Zs = []
        self.reals = []
        self.noiseAmp = []
        self.writer = None

    def init_models(self):
        generator = Generator(self.config).to(self.config.device)
        generator.apply(weights_init)
        if self.config.generator_path is not None:
            generator.load_state_dict(torch.load(self.config.generator_path))
        # print(generator)

        discriminator = Discriminator(self.config).to(self.config.device)
        discriminator.apply(weights_init)
        if self.config.discriminator_path is not None:
            discriminator.load_state_dict(torch.load(self.config.discriminator_path))
        # print(discriminator)

        return discriminator, generator

    def train(self):
        # Prepare image pyramid
        train_img = read_img(self.config)
        real = resize_img(train_img, self.config.start_scale, self.config)
        self.reals = creat_reals_pyramid(real, self.reals, self.config)

        prev_img = 0
        prev_nfc = 0
        self.config.out_ = generate_dir2save(self.config)
        self.writer = SummaryWriter(f'{self.config.out_}/logs')

        # Pyramid training
        for scale_iter in range(self.config.stop_scale+1):
            # Become larger as scale_iter increase (maximum=128)
            self.config.nfc = min(self.config.nfc_init * pow(2, math.floor(scale_iter / 4)), 128)
            self.config.min_nfc = min(self.config.min_nfc_init * pow(2, math.floor(scale_iter / 4)), 128)

            # Prepare directory to save images
            self.config.result_dir = f'{self.config.out_}/{scale_iter}'
            os.makedirs(self.config.result_dir, exist_ok=True)
            plt.imsave(f'{self.config.result_dir}/real_scale.png', torch2np(self.reals[scale_iter]), vmin=0, vmax=1)

            cur_discriminator, cur_generator = self.init_models()
            if prev_nfc == self.config.nfc:
                cur_generator.load_state_dict(torch.load(f'{self.config.out_}/{scale_iter - 1}/generator.pth'))
                cur_discriminator.load_state_dict(torch.load(f'{self.config.out_}/{scale_iter - 1}/discriminator.pth'))

            cur_z, prev_img, cur_generator = self.train_single_scale(cur_discriminator, cur_generator, prev_img)
            cur_generator = reset_grads(cur_generator, False)
            cur_generator.eval()
            cur_discriminator = reset_grads(cur_discriminator, False)
            cur_discriminator.eval()

            self.Gs.append(cur_generator)
            self.Zs.append(cur_z)
            self.noiseAmp.append(self.config.noise_amp)

            torch.save(self.Zs, f'{self.config.out_}/Zs.pth')
            torch.save(self.Gs, f'{self.config.out_}/Gs.pth')
            torch.save(self.reals, f'{self.config.out_}/reals.pth')
            torch.save(self.noiseAmp, f'{self.config.out_}/noiseAmp.pth')

            prev_nfc = self.config.nfc
            del cur_discriminator, cur_generator

        return

    def train_single_scale(self, cur_discriminator, cur_generator, prev_img):
        real = self.reals[len(self.Gs)]

        # Set padding layer(Initial padding) - To Do (Change this for noise padding not zero-padding)g
        self.config.z_h = real.shape[2]
        self.config.z_w = real.shape[3]
        self.config.receptive_field = self.config.kernel_size + ((self.config.kernel_size - 1) * (self.config.num_layers - 1)) * self.config.stride
        pad_noise = int(((self.config.kernel_size - 1) * self.config.num_layers) / 2)
        pad_image = int(((self.config.kernel_size - 1) * self.config.num_layers) / 2)
        m_noise = nn.ZeroPad2d(int(pad_noise))
        m_image = nn.ZeroPad2d(int(pad_image))

        # All zero except for the coarsest generator - To Do (Select z_opt well - like VAE?)
        z_opt = torch.full([1, self.config.num_z_channel, self.config.z_h, self.config.z_w], 0, dtype=torch.float32, device=self.config.device)
        z_opt = m_noise(z_opt)

        # MultiStepLR: lr *= gamma every time reaches one of the milestones
        D_optimizer = optim.Adam(cur_discriminator.parameters(), lr=self.config.d_lr, betas=(self.config.beta1, self.config.beta2))
        G_optimizer = optim.Adam(cur_generator.parameters(), lr=self.config.g_lr, betas=(self.config.beta1, self.config.beta2))
        D_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=D_optimizer, milestones=[1600], gamma=self.config.gamma)
        G_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=G_optimizer, milestones=[1600], gamma=self.config.gamma)

        for epoch in tqdm(range(self.config.num_iter), desc=f'{len(self.Gs)}th GAN'):
            # Basic noise setting
            if self.Gs == [] and self.config.mode != 'SR_train':    # Coarsest generator and not SR
                z_opt = generate_noise([1, self.config.z_h, self.config.z_w], device=self.config.device)    # fixed_noise
                z_opt = m_noise(z_opt.expand(1, 3, self.config.z_h, self.config.z_w))                       # Pad(Copy noise to 3 channels)
                noise_ = generate_noise([1, self.config.z_h, self.config.z_w], device=self.config.device)   # Noise to generate fake image - purely random
                noise_ = m_noise(noise_.expand(1, 3, self.config.z_h, self.config.z_w))
            else:                                                   # SR_train mode or not Coarsest generator
                noise_ = generate_noise([self.config.num_z_channel, self.config.z_h, self.config.z_w], device=self.config.device)   # Not copying noise into 3 channel
                noise_ = m_noise(noise_)

            # Train Discriminator: Maximize D(x) - D(G(z)) -> Minimize D(G(z)) - D(X)
            for i in range(self.config.n_critic):
                # Manage data
                if i == 0 and epoch == 0:                                   # Real first iteration
                    if self.Gs == [] and self.config.mode != 'SR_train':    # Coarsest generator, not SR
                        prev = torch.full([1, self.config.num_z_channel, self.config.z_h, self.config.z_w], 0, device=self.config.device)
                        prev_img = prev
                        prev = m_image(prev)
                        z_prev = torch.full([1, self.config.num_z_channel, self.config.z_h, self.config.z_w], 0, device=self.config.device)
                        z_prev = m_noise(z_prev)
                        self.config.noise_amp = 1
                    elif self.config.mode == 'SR_train':                    # SR
                        z_prev = prev_img
                        criterion = nn.MSELoss()                            # SR criterion
                        rmse = torch.sqrt(criterion(real, z_prev))
                        self.config.noise_amp = self.config.noise_amp_init * rmse
                        z_prev = m_image(z_prev)
                        prev = z_prev
                    else:                                                   # Not first generator, not SR
                        prev = self.draw_sequentially(prev_img, 'rand', m_noise, m_image)
                        prev = m_image(prev)
                        z_prev = self.draw_sequentially(prev_img, 'rec', m_noise, m_image)
                        criterion = nn.MSELoss()
                        rmse = torch.sqrt(criterion(real, z_prev))
                        self.config.noise_amp = self.config.noise_amp_init * rmse
                        z_prev = m_image(z_prev)
                else:                                                       # Not first training iteration
                    prev = self.draw_sequentially(prev_img, 'rand', m_noise, m_image)
                    prev = m_image(prev)

                if self.Gs == [] and self.config.mode != 'SR_train':        # First generator, not SR
                    noise = noise_
                else:                                                       # Not first generator or SR
                    noise = self.config.noise_amp * noise_ + prev

                # Train with real data
                cur_discriminator.zero_grad()
                real_prob_out = cur_discriminator(real)
                d_real_loss = -real_prob_out.mean()                         # Maximize D(X) -> Minimize -D(X)
                d_real_loss.backward(retain_graph=True)
                D_x = -d_real_loss.item()

                # Train with fake data
                fake = cur_generator(noise, prev)
                fake_prob_out = cur_discriminator(fake)
                d_fake_loss = fake_prob_out.mean()                          # Minimize D(G(z))
                d_fake_loss.backward(retain_graph=True)
                D_G_z = d_fake_loss.item()

                # Gradient penalty
                gradient_penalty = calcul_gp(cur_discriminator, real, fake, self.config.gp_lambda, self.config.device)
                gradient_penalty.backward()
                D_optimizer.step()

                d_loss = d_real_loss + d_fake_loss + gradient_penalty
                critic = D_x - D_G_z

            # Train Generator : Maximize D(G(z)) -> Minimize -D(G(z))
            for i in range(self.config.generator_iter):
                # Adversarial loss
                cur_generator.zero_grad()
                fake_prob_out = cur_discriminator(fake)
                g_adv_loss = -fake_prob_out.mean()
                g_adv_loss.backward(retain_graph=True)
                g_adv_loss = g_adv_loss.item()

                # Reconstruction loss
                if self.config.alpha != 0:
                    mse_criterion = nn.MSELoss()
                    Z_opt = self.config.noise_amp * z_opt + z_prev
                    g_rec_loss = self.config.alpha * mse_criterion(cur_generator(Z_opt.detach(), z_prev), real)
                    g_rec_loss.backward()
                    g_rec_loss = g_rec_loss.item()
                else:
                    Z_opt = z_opt
                    g_rec_loss = 0

                G_optimizer.step()
                g_loss = g_adv_loss + (self.config.alpha * g_rec_loss)

            # Log losses
            self.writer.add_scalar(f'{len(self.Gs)}th_G/total_loss', g_loss, epoch)
            self.writer.add_scalar(f'{len(self.Gs)}th_G/adv_loss', g_adv_loss, epoch)
            self.writer.add_scalar(f'{len(self.Gs)}th_G/recons_loss', g_rec_loss, epoch)
            self.writer.add_scalar(f'{len(self.Gs)}th_D/total_loss', d_loss, epoch)
            self.writer.add_scalar(f'{len(self.Gs)}th_D/critic', critic, epoch)
            self.writer.add_scalar(f'{len(self.Gs)}th_D/gp', gradient_penalty, epoch)

            # Log image
            if epoch % self.config.img_save_iter == 0 or epoch == (self.config.num_iter - 1):
                plt.imsave(f'{self.config.result_dir}/{epoch}_fake_sample.png', torch2np(fake.detach()), vmin=0, vmax=1)
                plt.imsave(f'{self.config.result_dir}/{epoch}_fixed_noise.png', torch2np(Z_opt.detach() * 2 - 1), vmin=0, vmax=1)
                plt.imsave(f'{self.config.result_dir}/{epoch}_reconstruction.png', torch2np(cur_generator(Z_opt.detach(), z_prev).detach()), vmin=0, vmax=1)

            D_scheduler.step()
            G_scheduler.step()

        # Save model weights
        torch.save(cur_generator.state_dict(), f'{self.config.result_dir}/generator.pth')
        torch.save(cur_discriminator.state_dict(), f'{self.config.result_dir}/discriminator.pth')
        torch.save(z_opt, f'{self.config.result_dir}/z_opt.pth')

        return z_opt, prev_img, cur_generator

    def draw_sequentially(self, in_s, mode, m_noise, m_image):
        G_z = in_s
        if len(self.Gs) > 0:
            if mode == 'rand':
                count = 0
                pad_noise = int(((self.config.kernel_size - 1) * self.config.num_layers) / 2)
                for G, Z_opt, real_curr, real_next, noise_amp in zip(self.Gs, self.Zs, self.reals, self.reals[1:], self.noiseAmp):
                    if count == 0:
                        z = generate_noise( [1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=self.config.device)
                        z = z.expand(1, 3, z.shape[2], z.shape[3])
                    else:
                        z = generate_noise([self.config.num_z_channel, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=self.config.device)
                    z = m_noise(z)
                    G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                    G_z = m_image(G_z)
                    z_in = noise_amp * z + G_z
                    G_z = G(z_in.detach(), G_z)
                    G_z = resize_img(G_z, 1 / self.config.scale_factor, self.config)
                    G_z = G_z[:, :, 0:real_next.shape[2], 0:real_next.shape[3]]
                    count += 1
            if mode == 'rec':
                count = 0
                for G, Z_opt, real_curr, real_next, noise_amp in zip(self.Gs, self.Zs, self.reals, self.reals[1:], self.noiseAmp):
                    G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                    G_z = m_image(G_z)
                    z_in = noise_amp * Z_opt + G_z
                    G_z = G(z_in.detach(), G_z)
                    G_z = resize_img(G_z, 1 / self.config.scale_factor, self.config)
                    G_z = G_z[:, :, 0:real_next.shape[2], 0:real_next.shape[3]]
                    count += 1
        return G_z

    # def draw_sequentially(self, prev_img, mode, m_noise, m_image):
    #     upscaled_prev = prev_img
    #     if len(self.Gs) > 0:
    #         if mode == 'rec':
    #             count = 0
    #             for G, Z_opt, cur_real, next_real, noise_amp in zip(self.Gs, self.Zs, self.reals, self.reals[1:], self.noiseAmp):
    #                 upscaled_prev = upscaled_prev[:, :, 0:cur_real.shape[2], 0:cur_real.shape[3]]
    #                 padded_img = m_image(upscaled_prev)
    #                 padded_img_with_noise = noise_amp * Z_opt + padded_img                          # Z_opt is already padded
    #                 generated_img = G(padded_img_with_noise.detach(), padded_img)
    #                 up_scaled_img = resize_img(generated_img, 1/self.config.scale_factor, self.config)
    #                 upscaled_prev = up_scaled_img[:, :, 0:next_real.shape[2], 0:next_real.shape[3]]
    #                 count += 1
    #         elif mode == 'rand':
    #             count = 0
    #             pad_noise = int(((self.config.kernel_size - 1) * self.config.num_layers) / 2)
    #             for G, Z_opt, cur_real, next_real, noise_amp in zip(self.Gs, self.Zs, self.reals, self.reals[1:], self.noiseAmp):
    #                 if count == 0:  # Generate random 1-channel noise
    #                     random_noise = generate_noise([1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=self.config.device)
    #                     random_noise = random_noise.expand(1, 3, random_noise.shape[2], random_noise.shape[3])
    #                 else:           # Generate random 3-channel noise
    #                     random_noise = generate_noise([self.config.num_z_channel, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=self.config.device)
    #                 padded_noise = m_noise(random_noise)
    #                 upscaled_prev = upscaled_prev[:, :, 0:cur_real.shape[2], 0:cur_real.shape[3]]
    #                 padded_img = m_image(upscaled_prev)
    #                 padded_img_with_noise = noise_amp * padded_noise + padded_img
    #                 generated_img = G(padded_img_with_noise.detach(), padded_img)
    #                 up_scaled_img = resize_img(generated_img, 1/self.config.scale_factor, self.config)
    #                 upscaled_prev = up_scaled_img[:, :, 0:next_real.shape[2], 0:next_real.shape[3]]
    #                 count += 1
    #     return upscaled_prev


