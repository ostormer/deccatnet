import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce


class Projector(nn.Module):
    def __init__(self, params):
        for key in params:
            self.key = params[key]

    def forward(self, x_in):
        pass

class Transformer(nn.Module):
    def __init__(self,params):
        super().__init__()
        for key in params:
            self.key = params[key]

    def forward(self):
        pass

class Convolution(nn.Module):
    def __init__(self, params):
        # self.patch_size = patch_size
        super().__init__()
        for key in params:
            self.key = params[key]

        # self.kernel_size_1 = (1,25) and self.kernel_size_2 = (22,1), self.drop_r = 0.5, self.pool_kernel, self.pool_stirde = (1, 75), (1, 15)
        self.shallownet = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_size, self.kernel_size_1, (1, 1)),
            nn.Conv2d(self.hidden_size, self.hidden_size, self.kernel_size_2, (1, 1)),
            nn.BatchNorm2d(self.hidden_size),
            nn.ELU(),
            nn.AvgPool2d(self.pool_kernel, self.pool_stride),
            # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(self.drop_r),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, self.emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x

class Encoder(nn.Sequential):
    def __init__(self, conv_params, trans_params):
        self.conv = Convolution(conv_params)
        self.trans = Transformer(trans_params)

    def forward(self, x_in):
        x = self.conv(x_in)
        x_out = self.trans(x)
        return  x_out

class TransPreTrain(nn.Module):
    def __init__(self, conv_params,trans_params, project_params):
        self.encoder = Encoder(conv_params, trans_params)
        self.projector = Projector(project_params)

    def forward(self, x_in):
        x = self.encoder(x_in)
        x_out = self.projector(x)
        return x_out
