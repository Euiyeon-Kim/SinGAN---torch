import torch
import torch.nn as nn

from utils.layers import ConvBlock
from model.modules.custom_acm import CustomACM


class ACMDiscriminator(nn.Module):
    def __init__(self, config, num_heads, real):
        super(ACMDiscriminator, self).__init__()
        self.ans = real
        self.config = config
        self.is_cuda = torch.cuda.is_available()

        N = int(config.nfc)
        self.head = ConvBlock(config.img_channel, N, config.kernel_size, 1, config.pad)
        self.body = nn.Sequential()
        for i in range(config.num_layers - 2):
            N = int(config.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, config.min_nfc), max(N, config.min_nfc), config.kernel_size, 1, config.pad)
            self.body.add_module('block%d' % (i + 1), block)
        self.acm = CustomACM(num_heads=num_heads, num_features=max(N, config.min_nfc), orthogonal_loss=self.config.use_acm_oth)
        # WGAN-GP discriminator has no activation at last layer
        self.tail = nn.Conv2d(max(N, config.min_nfc), 1, kernel_size=config.kernel_size, stride=1, padding=config.pad)

    def forward(self, x):
        x_feature = self.head(x)
        ans_feature = self.head(self.ans)

        x_feature = self.body(x_feature)
        ans_feature = self.body(ans_feature)

        x, oth, add_att_maps, sub_att_maps = self.acm(x_feature, ans_feature)
        x = self.tail(x)
        return x, oth, add_att_maps, sub_att_maps
