import os
from shutil import copyfile

from config import Config
from model.SinGAN import SinGAN
from model.ACM_SinGAN import SinGAN_ACM
from utils.image import read_img
from utils.utils import process_config, adjust_scales, calcul_sr_scale

# All data : [B C H W]

if __name__ == '__main__':
    os.makedirs(Config.exp_dir, exist_ok=True)
    copyfile('config.py', f'{Config.exp_dir}/config.py')
    process_config(Config)

    train_img = read_img(Config)
    if Config.mode == "train":
        adjust_scales(train_img, Config)
    elif Config.mode == "train_SR":
        calcul_sr_scale(Config)
        adjust_scales(train_img, Config)

    singan = SinGAN_ACM(config=Config) if Config.use_acm else SinGAN(config=Config)
    singan.train()
