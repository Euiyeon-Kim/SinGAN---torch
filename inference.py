import os

from config import Config
from model.SinGAN import SinGAN
from utils.image import read_img
from utils.utils import generate_dir2save, process_config, adjust_scales

gen_start_scale = 0
scale_h = 1.5
scale_w = 0.7
num_samples = 10

if __name__ == '__main__':
    process_config(Config)
    inference_img = read_img(Config)
    adjust_scales(inference_img, Config)

    Config.exp_dir = generate_dir2save(Config)
    Config.infer_dir = f'{Config.exp_dir}/infer'
    os.makedirs(Config.infer_dir, exist_ok=True)

    singan = SinGAN(config=Config)
    singan.load_trained_weights()
    start_img_input = singan.create_inference_input(gen_start_scale, scale_h, scale_w)
    singan.inference(gen_start_scale, start_img_input, scale_h, scale_w, num_samples)
