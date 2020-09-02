import os
import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model.generator import Generator
from model.discriminator import Discriminator
from utils.layers import weights_init
from utils.image import read_img, resize_img, torch2np
from utils.utils import creat_reals_pyramid, generate_dir2save, generate_noise, calcul_gp


class SinGAN:
    def __init__(self, config):
        self.config = config
        self.Gs = []
        self.Zs = []
        self.reals = []
        self.noiseAmp = []
        self.writer = SummaryWriter()

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
        train_img = read_img(self.config)
        real = resize_img(train_img, self.config.start_scale, self.config)
        self.reals = creat_reals_pyramid(real, self.reals, self.config)

        prev_img = 0
        prev_nfc = 0

        for scale_iter in range(self.config.stop_scale+1):
            # Become larger as scale_iter increase (maximum=128)
            self.config.nfc = min(self.config.nfc_init * pow(2, math.floor(scale_iter / 4)), 128)
            self.config.min_nfc = min(self.config.min_nfc_init * pow(2, math.floor(scale_iter / 4)), 128)

            # Prepare directory to save images
            self.config.out_ = generate_dir2save(self.config)
            self.config.result_dir = f'{self.config.out_}/{scale_iter}'
            os.makedirs(self.config.result_dir, exist_ok=True)
            plt.imsave(f'{self.config.result_dir}/real_scale.png', torch2np(self.reals[scale_iter]), vmin=0, vmax=1)

            cur_discriminator, cur_generator = self.init_models()
            if prev_nfc == self.config.nfc:
                cur_generator.load_state_dict(torch.load(f'{self.config.out_}/{scale_iter - 1}/generator.pth'))
                cur_discriminator.load_state_dict(torch.load(f'{self.config.out_}/{scale_iter - 1}/discriminator.pth'))

            cur_z, prev_img, cur_generator = self.train_single_scale(cur_discriminator, cur_generator, prev_img)

    def train_single_scale(self, cur_discriminator, cur_generator, prev_img):
        real = self.reals[len(self.Gs)]

        # Set noise dimension
        self.config.z_h = real.shape[2]
        self.config.z_w = real.shape[3]
        self.config.receptive_field = self.config.kernel_size + ((self.config.kernel_size - 1) * (self.config.num_layers - 1)) * self.config.stride
        pad_noise = int(((self.config.kernel_size - 1) * self.config.num_layers) / 2)
        pad_image = int(((self.config.kernel_size - 1) * self.config.num_layers) / 2)

        # Padding layer - To Do (Change this for noise padding not zero-padding)
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

        for epoch in range(self.config.num_iter):
            # Basic noise setting
            if self.Gs == [] and self.config.mode != 'SR_train':    # Coarsest generator and not SR
                z_opt = generate_noise([1, self.config.z_h, self.config.z_w], device=self.config.device)    # fixed_noise
                z_opt = m_noise(z_opt.expand(1, 3, self.config.z_h, self.config.z_w))                       # Copy noise to 3 channels
                noise_ = generate_noise([1, self.config.z_h, self.config.z_w], device=self.config.device)   # Noise to generate fake image - purely random
                noise_ = m_noise(noise_.expand(1, 3, self.config.z_h, self.config.z_w))
            else:       # SR_train mode or not first generator
                noise_ = generate_noise([self.config.num_z_channel, self.config.z_h, self.config.z_w], device=self.config.device)   # Not copying noise into 3 channel
                noise_ = m_noise(noise_)

            # Train Discriminator: Minimize D(G(z)) - D(X)
            for i in range(self.config.n_critic):
                # Manage data
                if i == 0 and epoch == 0:                                   # Real first iteration
                    if self.Gs == [] and self.config.mode != 'SR_train':    # First generator, not SR
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
                        prev = self.draw_concat(prev_img, 'rand', m_noise, m_image)
                else:                                                       # Not first training iteration
                    prev = self.draw_concat(prev_img, 'rand', m_noise, m_image)
                    prev = m_image(prev)

                if self.Gs == [] and self.config.mode != 'SR_train':        # First generator, not SR
                    noise = noise_
                else:                                                       # Not first generator or SR
                    noise = self.config.noise_amp * noise_ + prev

                # Train with real data
                cur_discriminator.zero_grad()
                real_prob_out = cur_discriminator(real).to(self.config.device)

                real_wasserstein_loss = -real_prob_out.mean()               # Maximize D(X) -> Minimize -D(X)
                real_wasserstein_loss.backward(retain_graph=True)
                d_loss_real = -real_wasserstein_loss.item()

                # Train with fake data
                fake = cur_generator(noise.detach(), prev)
                fake_prob_out = cur_discriminator(fake.detach())
                fake_wasserstein_loss = fake_prob_out.mean()                # Minimize D(G(z))
                fake_wasserstein_loss.backward(retain_graph=True)
                d_loss_fake = fake_prob_out.mean().item()

                # Gradient penalty
                gradient_penalty = calcul_gp(cur_discriminator, real, fake, self.config.gp_lambda, self.config.device)
                gradient_penalty.backward()
                d_loss = real_wasserstein_loss + fake_wasserstein_loss + gradient_penalty
                D_optimizer.step()

            # Train Generator : Maximize D(G(z))
            # for i in range(self.config.generator_iter):
            #     print(torch.sum(fake))
            #     exit()
            #     cur_generator.zero_grad()

    def draw_concat(self, prev_img, mode, m_noise, m_image):
        G_z = prev_img
        return G_z


