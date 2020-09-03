import os
import math
import random

import torch
import torch.nn as nn
from utils.image import resize_img


def process_config(config):
    if config.manualSeed is None:
        config.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", config.manualSeed)
    random.seed(config.manualSeed)
    torch.manual_seed(config.manualSeed)

    # Device parameters
    config.useGPU = torch.cuda.is_available()
    config.device = torch.device('cuda') if config.useGPU else torch.device('cpu')
    # Pyramid parameters
    config.noise_amp_init = config.noise_amp
    config.scale_factor_init = config.scale_factor
    # Network parameters
    config.nfc_init = config.nfc
    config.min_nfc_init = config.min_nfc
    # Save options
    config.out_ = 'exps/%s/scale_factor=%f/' % (os.path.basename(config.img_path)[:-4], config.scale_factor)


def adjust_scales(real, config):
    minwh = min(real.shape[2], real.shape[3])
    maxwh = max(real.shape[2], real.shape[3])
    config.num_scales = math.ceil(math.log(config.min_size / minwh, config.scale_factor_init)) + 1
    scale2stop = math.ceil(math.log(min([config.max_size, maxwh]) / maxwh, config.scale_factor_init))
    config.stop_scale = config.num_scales - scale2stop
    config.start_scale = min(config.max_size / maxwh, 1)
    resized_real = resize_img(real, config.start_scale, config)
    config.scale_factor = math.pow(config.min_size/min(resized_real.shape[2], resized_real.shape[3]), 1/config.stop_scale)
    scale2stop = math.ceil(math.log(min([config.max_size, maxwh]) / maxwh, config.scale_factor_init))
    config.stop_scale = config.num_scales - scale2stop
    return resized_real


# To Do - Improve making pyramid process
def creat_reals_pyramid(real, reals, config):
    for i in range(config.stop_scale+1):
        scale = math.pow(config.scale_factor, config.stop_scale-i)
        curr_real = resize_img(real, scale, config)
        reals.append(curr_real)
    return reals


def generate_dir2save(config):
    dir2save = None
    if (config.mode == 'train') | (config.mode == 'SR_train'):
        dir2save = f'exp/{os.path.basename(config.img_path)[:-4]}/scale-{config.scale_factor_init}_alp-{config.alpha}'
    return dir2save


def upsampling(img, sx, sy):
    m = nn.Upsample(size=[round(sx), round(sy)], mode='bilinear', align_corners=True)
    return m(img)


def generate_noise(size, num_noise=1, device='cuda', type='gaussian', scale=1):
    if type == 'gaussian':
        noise = torch.randn(num_noise, size[0], round(size[1]/scale), round(size[2]/scale), device=device)
        noise = upsampling(noise, size[1], size[2])
    elif type =='gaussian_mixture':
        noise1 = torch.randn(num_noise, size[0], size[1], size[2], device=device) + 5
        noise2 = torch.randn(num_noise, size[0], size[1], size[2], device=device)
        noise = noise1 + noise2
    elif type == 'uniform':
        noise = torch.randn(num_noise, size[0], size[1], size[2], device=device)
    else:
        raise Exception('Unimplemented noise type')
    return noise
