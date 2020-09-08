import torch
import torch.nn as nn

from utils.layers import ConvBlock


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(config.nfc)

        self.head = ConvBlock(config.img_channel, N, config.kernel_size, 1, config.pad)

        self.body = nn.Sequential()
        for i in range(config.num_layers - 2):
            N = int(config.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, config.min_nfc), max(N, config.min_nfc), config.kernel_size, 1, config.pad)
            self.body.add_module('block%d' % (i + 1), block)

        # WGAN-GP discriminator has no activation at last layer
        self.tail = nn.Conv2d(max(N, config.min_nfc), 1, kernel_size=config.kernel_size, stride=1, padding=config.pad)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x

    def feature(self, x):
        x = self.head(x)
        x = self.body(x)
        return x