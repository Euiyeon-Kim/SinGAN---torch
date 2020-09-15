import os

import matplotlib.pyplot as plt

from config import Config
# from model.ACM_SinGAN import SinGAN_ACM as SinGAN
from model.SinGAN import SinGAN
from utils.image import read_img, torch2np
from utils.utils import process_config, adjust_scales, calcul_sr_scale


if __name__ == '__main__':
    process_config(Config)
    Config.infer_dir = f'{Config.exp_dir}/infer_pyramid' if Config.save_all_pyramid else f'{Config.exp_dir}/infer'
    singan = SinGAN(config=Config)
    singan.load_trained_weights()

    inference_img = read_img(Config)
    os.makedirs(Config.infer_dir, exist_ok=True)
    plt.imsave(f'{Config.exp_dir}/real.png', torch2np(inference_img), vmin=0, vmax=1)

    if Config.mode == 'train':
        adjust_scales(inference_img, Config)
        start_img_input = singan.create_inference_input()
    elif Config.mode == 'train_SR':
        in_scale, iter_num = calcul_sr_scale(Config)
        Config.scale_factor = 1 / in_scale
        Config.scale_factor_init = 1 / in_scale
        Config.scale_w = 1
        Config.scale_h = 1
        start_img_input = singan.create_sr_inference_input(singan.reals[-1], iter_num)

    out = singan.inference(start_img_input)
    if Config.mode == 'train_SR':
        out = out[:, :, 0:int(Config.sr_factor * singan.reals[-1].shape[2]), 0:int(Config.sr_factor * singan.reals[-1].shape[3])]
        plt.imsave(f'{Config.exp_dir}/sr.png', torch2np(out), vmin=0, vmax=1)

