import torch
import torch.nn as nn

from utils.layers import ConvBlock


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = config.nfc

        self.head = ConvBlock(config.nc_im, N, config.kernel_size, 1, config.pad)

        self.body = nn.Sequential()
        for i in range(config.num_layers - 2):
            N = int(config.nfc / pow(2, (i+1)))
            block = ConvBlock(max(2*N, config.min_nfc), max(N, config.min_nfc), config.kernel_size, 1, config.pad)
            self.body.add_module('block%d' % (i+1), block)

        self.tail = nn.Sequential(
            nn.Conv2d(max(N,config.min_nfc), config.nc_im, kernel_size=config.kernel_size, stride =1, padding=config.pad),
            nn.Tanh()
        )

    def forward(self, x, y):    # x:noise, y:prev
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        # To Do (이게 대체 무엇인가,,,)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y
