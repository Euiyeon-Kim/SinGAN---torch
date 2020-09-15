import math
import random
import multiprocessing

import torch
import torch.nn as nn
from utils.image import resize_img


def process_config(config):
    # Setting seeds
    if config.manualSeed is None:
        config.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", config.manualSeed)
    random.seed(config.manualSeed)
    torch.manual_seed(config.manualSeed)
    # Multiprocessing
    config.num_cores = multiprocessing.cpu_count()
    # Device parameters
    config.useGPU = torch.cuda.is_available()
    config.device = torch.device('cuda') if config.useGPU else torch.device('cpu')
    # Pyramid parameters
    config.noise_amp_init = config.noise_amp
    config.scale_factor_init = config.scale_factor
    # Network parameters
    config.nfc_init = config.nfc
    config.min_nfc_init = config.min_nfc
    # sr
    if config.mode == 'train_SR':
        config.alpha = 100


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


def calcul_sr_scale(config):
    in_scale = math.pow(1/2, 1/3)
    iter_num = round(math.log(1/config.sr_factor, in_scale))
    in_scale = pow(config.sr_factor, 1 / iter_num)
    config.scale_factor = 1 / in_scale
    config.scale_factor_init = 1 / in_scale
    config.min_size = 18
    return in_scale, iter_num


# To Do - Improve making pyramid process
def creat_reals_pyramid(real, reals, config):
    for i in range(config.stop_scale+1):
        scale = math.pow(config.scale_factor, config.stop_scale-i)
        curr_real = resize_img(real, scale, config)
        reals.append(curr_real)
    return reals


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
