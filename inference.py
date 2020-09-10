import os

import matplotlib.pyplot as plt

from config import Config
from model.SinGAN import SinGAN
from utils.image import read_img, torch2np
from utils.utils import process_config, adjust_scales

use_fixed_noise = False
save_all_pyramid = True
gen_start_scale = 0
scale_h = 2
scale_w = 2
num_samples = 5

if __name__ == '__main__':
    process_config(Config)
    inference_img = read_img(Config)
    adjust_scales(inference_img, Config)

    Config.use_fixed_noise = use_fixed_noise
    Config.save_all_pyramid = save_all_pyramid
    Config.infer_dir = f'{Config.exp_dir}/infer_pyramid' if save_all_pyramid else f'{Config.exp_dir}/infer'
    os.makedirs(Config.infer_dir, exist_ok=True)
    plt.imsave(f'{Config.exp_dir}/real.png', torch2np(inference_img), vmin=0, vmax=1)

    singan = SinGAN(config=Config)
    singan.load_trained_weights()
    start_img_input = singan.create_inference_input(gen_start_scale, scale_h, scale_w)
    singan.inference(gen_start_scale, start_img_input, scale_h, scale_w, num_samples)
