import os
from shutil import copyfile

from config import Config
from model.ACM_SinGAN import SinGAN_ACM as SinGAN
from utils.image import read_img
from utils.utils import process_config, adjust_scales

# All data : [B C H W]

if __name__ == '__main__':
    os.makedirs(Config.exp_dir, exist_ok=True)
    copyfile('config.py', f'{Config.exp_dir}/config.py')
    process_config(Config)
    train_img = read_img(Config)
    adjust_scales(train_img, Config)

    singan = SinGAN(config=Config)
    singan.train()
