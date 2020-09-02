from config import Config
from model.SinGAN import SinGAN
from utils.image import read_img
from utils.utils import process_config, adjust_scales

# All data : [B C H W]

if __name__ == '__main__':
    process_config(Config)
    train_img = read_img(Config)
    adjust_scales(train_img, Config)

    singan = SinGAN(config=Config)
    singan.train()
