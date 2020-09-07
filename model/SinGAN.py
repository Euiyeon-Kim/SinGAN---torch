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
        self.noise_amps = []
        self.writer = None
        self.first_img_input = None
        self.log_losses = {}

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

        prev_nfc = 0
        self.config.exp_dir = generate_dir2save(self.config)
        self.writer = SummaryWriter(f'{self.config.exp_dir}/logs')

        # Pyramid training
        for scale_iter in range(self.config.stop_scale+1):
            # Become larger as scale_iter increase (maximum=128)
            self.config.nfc = min(self.config.nfc_init * pow(2, math.floor(scale_iter / 4)), 128)
            self.config.min_nfc = min(self.config.min_nfc_init * pow(2, math.floor(scale_iter / 4)), 128)

            # Prepare directory to save images
            self.config.result_dir = f'{self.config.exp_dir}/{scale_iter}'
            os.makedirs(self.config.result_dir, exist_ok=True)
            plt.imsave(f'{self.config.result_dir}/real_scale.png', torch2np(self.reals[scale_iter]), vmin=0, vmax=1)

            cur_discriminator, cur_generator = self.init_models()
            if prev_nfc == self.config.nfc:
                cur_generator.load_state_dict(torch.load(f'{self.config.exp_dir}/{scale_iter - 1}/generator.pth'))
                cur_discriminator.load_state_dict(torch.load(f'{self.config.exp_dir}/{scale_iter - 1}/discriminator.pth'))

            cur_z, cur_generator = self.train_single_stage(cur_discriminator, cur_generator)
            cur_generator = reset_grads(cur_generator, False)
            cur_generator.eval()
            cur_discriminator = reset_grads(cur_discriminator, False)
            cur_discriminator.eval()

            self.Gs.append(cur_generator)
            self.Zs.append(cur_z)
            self.noise_amps.append(self.config.noise_amp)

            torch.save(self.Zs, f'{self.config.exp_dir}/Zs.pth')
            torch.save(self.Gs, f'{self.config.exp_dir}/Gs.pth')
            torch.save(self.reals, f'{self.config.exp_dir}/reals.pth')
            torch.save(self.noise_amps, f'{self.config.exp_dir}/noiseAmp.pth')

            prev_nfc = self.config.nfc
            del cur_discriminator, cur_generator

        return

    def train_single_stage(self, cur_discriminator, cur_generator):
        real = self.reals[len(self.Gs)]
        _, _, real_h, real_w = real.shape

        # Set padding layer(Initial padding) - To Do (Change this for noise padding not zero-padding)g
        self.config.receptive_field = self.config.kernel_size + ((self.config.kernel_size - 1) * (self.config.num_layers - 1)) * self.config.stride
        padding_size = int(((self.config.kernel_size - 1) * self.config.num_layers) / 2)
        noise_pad = nn.ZeroPad2d(int(padding_size))
        image_pad = nn.ZeroPad2d(int(padding_size))

        # MultiStepLR: lr *= gamma every time reaches one of the milestones
        D_optimizer = optim.Adam(cur_discriminator.parameters(), lr=self.config.d_lr, betas=(self.config.beta1, self.config.beta2))
        G_optimizer = optim.Adam(cur_generator.parameters(), lr=self.config.g_lr, betas=(self.config.beta1, self.config.beta2))
        D_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=D_optimizer, milestones=[1600], gamma=self.config.gamma)
        G_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=G_optimizer, milestones=[1600], gamma=self.config.gamma)

        # Calculate noise amp(amount of info to generate) and recover prev_rec image
        if not self.Gs:
            rec_z = generate_noise([1, real_h, real_w], device=self.config.device).expand(1, 3, real_h, real_w)
            self.first_img_input = torch.full([1, self.config.img_channel, real_h, real_w], 0, device=self.config.device)
            upscaled_prev_rec_img = self.first_img_input
            self.config.noise_amp = 1
        else:
            rec_z = torch.full([1, self.config.img_channel, real_h, real_w], 0, device=self.config.device)
            upscaled_prev_rec_img = self.draw_sequentially('rec', noise_pad, image_pad)
            criterion = nn.MSELoss()
            rmse = torch.sqrt(criterion(real, upscaled_prev_rec_img))
            self.config.noise_amp = self.config.noise_amp_init * rmse
        padded_rec_z = noise_pad(rec_z)
        padded_rec_img = image_pad(upscaled_prev_rec_img)

        for epoch in tqdm(range(self.config.num_iter), desc=f'{len(self.Gs)}th GAN'):
            # Make noise input
            if not self.Gs:                                         # Generate fixed reconstruction noise
                random_z = generate_noise([1, real_h, real_w], device=self.config.device).expand(1, 3, real_h, real_w)
            else:                                                   # Reconstruction noise: Filled 0
                random_z = generate_noise([self.config.img_channel, real_h, real_w], device=self.config.device)
            padded_random_z = noise_pad(random_z)

            # Train Discriminator: Maximize D(x) - D(G(z)) -> Minimize D(G(z)) - D(X)
            for i in range(self.config.n_critic):
                # Make random image input
                upscaled_prev_random_img = self.draw_sequentially('rand', noise_pad, image_pad)
                padded_random_img = image_pad(upscaled_prev_random_img)
                padded_random_img_with_z = self.config.noise_amp * padded_random_z + padded_random_img

                # Train with real data
                cur_discriminator.zero_grad()
                real_prob_out = cur_discriminator(real)
                d_real_loss = -real_prob_out.mean()                         # Maximize D(X) -> Minimize -D(X)
                d_real_loss.backward()
                D_x = -d_real_loss.item()

                # Train with fake data
                fake = cur_generator(padded_random_img_with_z.detach(), padded_random_img)
                fake_prob_out = cur_discriminator(fake.detach())
                d_fake_loss = fake_prob_out.mean()                          # Minimize D(G(z))
                d_fake_loss.backward()
                D_G_z = d_fake_loss.item()

                # Gradient penalty
                gradient_penalty = calcul_gp(cur_discriminator, real, fake, self.config.gp_lambda, self.config.device)
                gradient_penalty.backward()

                D_optimizer.step()
                d_loss = d_real_loss + d_fake_loss + gradient_penalty
                critic = D_x - D_G_z
                self.log_losses[f'{len(self.Gs)}th_D/d'] = d_loss
                self.log_losses[f'{len(self.Gs)}th_D/d_critic'] = critic
                self.log_losses[f'{len(self.Gs)}th_D/d_gp'] = gradient_penalty

            # Train Generator : Maximize D(G(z)) -> Minimize -D(G(z))
            for i in range(self.config.generator_iter):
                cur_generator.zero_grad()

                # Make fake sample for every iteration
                upscaled_prev_random_img = self.draw_sequentially('rand', noise_pad, image_pad)
                padded_random_img = image_pad(upscaled_prev_random_img)
                padded_random_img_with_z = self.config.noise_amp * padded_random_z + padded_random_img
                fake = cur_generator(padded_random_img_with_z.detach(), padded_random_img)

                # Adversarial loss
                fake_prob_out = cur_discriminator(fake)
                g_adv_loss = -fake_prob_out.mean()
                g_adv_loss.backward()
                g_adv_loss = g_adv_loss.item()

                # Reconstruction loss
                mse_criterion = nn.MSELoss()
                padded_rec_img_with_z = self.config.noise_amp * padded_rec_z + padded_rec_img
                g_rec_loss = self.config.alpha * mse_criterion(cur_generator(padded_rec_img_with_z.detach(), padded_rec_img), real)
                g_rec_loss.backward()
                g_rec_loss = g_rec_loss.item()

                G_optimizer.step()
                g_loss = g_adv_loss + (self.config.alpha * g_rec_loss)
                self.log_losses[f'{len(self.Gs)}th_G/g'] = g_loss
                self.log_losses[f'{len(self.Gs)}th_G/g_critic'] = -g_adv_loss
                self.log_losses[f'{len(self.Gs)}th_G/g_rec'] = g_rec_loss

            # Log losses
            for key, value in self.log_losses.items():
                self.writer.add_scalar(key, value, epoch)

            # Log image
            if epoch % self.config.img_save_iter == 0 or epoch == (self.config.num_iter - 1):
                plt.imsave(f'{self.config.result_dir}/{epoch}_fake_sample.png', torch2np(fake.detach()), vmin=0, vmax=1)
                plt.imsave(f'{self.config.result_dir}/{epoch}_fixed_noise.png', torch2np(padded_rec_img_with_z.detach() * 2 - 1), vmin=0, vmax=1)
                plt.imsave(f'{self.config.result_dir}/{epoch}_reconstruction.png', torch2np(cur_generator(padded_rec_img_with_z.detach(), padded_rec_img).detach()), vmin=0, vmax=1)

            D_scheduler.step()
            G_scheduler.step()

        # Save model weights
        torch.save(cur_generator.state_dict(), f'{self.config.result_dir}/generator.pth')
        torch.save(cur_discriminator.state_dict(), f'{self.config.result_dir}/discriminator.pth')

        return padded_rec_z, cur_generator

    def draw_sequentially(self, mode, m_noise, m_image):
        upscaled_prev = self.first_img_input
        if len(self.Gs) > 0:
            if mode == 'rec':
                count = 0
                for G, Z_opt, cur_real, next_real, noise_amp in zip(self.Gs, self.Zs, self.reals, self.reals[1:], self.noise_amps):
                    upscaled_prev = upscaled_prev[:, :, 0:cur_real.shape[2], 0:cur_real.shape[3]]
                    padded_img = m_image(upscaled_prev)
                    padded_img_with_z = noise_amp * Z_opt + padded_img
                    generated_img = G(padded_img_with_z.detach(), padded_img)
                    up_scaled_img = resize_img(generated_img, 1/self.config.scale_factor, self.config)
                    upscaled_prev = up_scaled_img[:, :, 0:next_real.shape[2], 0:next_real.shape[3]]
                    count += 1
            elif mode == 'rand':
                count = 0
                pad_noise = int(((self.config.kernel_size - 1) * self.config.num_layers) / 2)
                for G, Z_opt, cur_real, next_real, noise_amp in zip(self.Gs, self.Zs, self.reals, self.reals[1:], self.noise_amps):
                    if count == 0:  # Generate random 1-channel noise
                        random_noise = generate_noise([1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=self.config.device)
                        random_noise = random_noise.expand(1, 3, random_noise.shape[2], random_noise.shape[3])
                    else:           # Generate random 3-channel noise
                        random_noise = generate_noise([self.config.img_channel, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=self.config.device)
                    padded_noise = m_noise(random_noise)
                    upscaled_prev = upscaled_prev[:, :, 0:cur_real.shape[2], 0:cur_real.shape[3]]
                    padded_img = m_image(upscaled_prev)
                    padded_img_with_z = noise_amp * padded_noise + padded_img
                    generated_img = G(padded_img_with_z.detach(), padded_img)
                    up_scaled_img = resize_img(generated_img, 1/self.config.scale_factor, self.config)
                    upscaled_prev = up_scaled_img[:, :, 0:next_real.shape[2], 0:next_real.shape[3]]
                    count += 1
        return upscaled_prev


