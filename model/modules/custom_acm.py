import torch
import torch.nn as nn

from model.modules.acm_module import AttendModule


class CustomACM(nn.Module):
    def __init__(self, num_heads, num_features, orthogonal_loss=True):
        super(CustomACM, self).__init__()
        assert num_features % num_heads == 0

        self.orthogonal_loss = orthogonal_loss
        self.num_features = num_features
        self.num_heads = num_heads

        self.att_mod = AttendModule(self.num_features, num_heads=num_heads)
        # self.sub_mod = AttendModule(self.num_features, num_heads=num_heads)

        self.init_parameters()

    def init_parameters(self):
        if self.att_mod is not None:
            self.att_mod.init_parameters()
        # if self.sub_mod is not None:
        #     self.sub_mod.init_parameters()

    def forward(self, x, ans):
        x_mu = x.mean([2, 3], keepdim=True)
        normalized_x = x - x_mu

        ans_mu = ans.mean([2, 3], keepdim=True)
        normalized_ans = ans - ans_mu

        # creates add or sub feature
        add_feature, add_att_maps = self.att_mod(normalized_x)         # K
        # creates add or sub feature
        sub_feature, sub_att_maps = self.att_mod(normalized_ans)       # Q

        y = (x + add_feature - sub_feature)
        if self.orthogonal_loss:
            oth = torch.mean(add_feature * sub_feature, dim=1, keepdim=True)
            return y, oth, add_att_maps, sub_att_maps
        else:
            return y, 0, add_att_maps, sub_att_maps
