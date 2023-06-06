import torch
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce

class DECCaTNet(nn.Module):
    def __init__(self, all_params, global_params):
        super().__init__()
        encoder_params = all_params['encoder_params']
        self.global_params = global_params
        self.encoder = Encoder(encoder_params, global_params)
        self.projector = Projector(encoder_params,global_params)

    def forward(self, x):
        x = self.encoder(x)
        #print(f'shape is {x.shape} and number of channels is {self.global_params["n_channels"]}')
        x = self.projector(x)
        return x


class Encoder(nn.Module):
    def __init__(self, encoder_params,global_params):
        super().__init__()
        self.ConvEmbedding = Convolution(encoder_params,global_params)
        self.TransEncoder = TransEncoder(encoder_params,global_params)

    def forward(self, x):
        x = self.ConvEmbedding(x)
        return self.TransEncoder(x)


class Convolution(nn.Module):
    """
    A traditional Transformer uses positional embedding to capture local features
    However we will use a CNN instead. This is due to the discovered effects
    in literature in combining CNN's and transformers for EEG classification
    """

    def __init__(self, encoder_params,global_params):
        super().__init__()

        temporal = encoder_params['temporal_size']
        spatial = global_params['n_channels']
        #encoder_params['spatial_size']
        dropout = encoder_params['CNN_dropout']
        self.emb_size = global_params['embedding_size']
        self.n_channels = global_params['n_channels']

        self.temporal = nn.Sequential(nn.Conv2d(1,temporal, kernel_size=(1, 25), stride=(1, 1))
                                      )
        self.spatial = nn.Sequential(nn.Conv2d(temporal, spatial, kernel_size=(self.n_channels, 1), stride=(1, 1)))
        self.pooling = nn.Sequential(nn.BatchNorm2d(spatial),
                                     nn.ELU(),
                                     nn.AvgPool2d((1, 75), (1, 15))
                                     )
        self.dropout = nn.Dropout(dropout)

        self.projector = nn.Sequential(nn.Conv2d(spatial, self.emb_size, kernel_size=(1, 1), stride=(1, 1)))
        self.rearrange = Rearrange('b e (h) (w) -> b (h w) e')
        # self.spatial = nn.Sequential(
        #
        # )
        # self.projector = nn.Sequential(
        #
        # )

    def forward(self, x):
        #x = torch.squeeze(x,dim=1)
        #print(f'first shape {x.shape}')
        x = self.temporal(x)
        x = self.spatial(x)
        #print(f'after temporal and spatial {x.shape}')
        x = self.pooling(x)
        #print(f'after pooling {x.shape}')
        x = self.dropout(x)
        x = self.projector(x)
        #print(f'after projector shape {x.shape}')
        x = self.rearrange(x)
        #print(f'output shape from Convolution {x.shape}') # this is what we essentially need.

        return x


class TransEncoder(nn.Module):
    def __init__(self, encoder_params,global_params):
        super().__init__()
        self.emb_size = global_params['embedding_size']
        self.n_heads = encoder_params['n_encoder_heads']
        self.n_layers = encoder_params['n_encoder_layers']

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.emb_size, nhead=self.n_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)

    def forward(self, x):
        x = self.encoder(x)
        return x


class Projector(nn.Module):
    def __init__(self, encoder_params,global_params):
        super().__init__()
        self.magic = global_params['magic_constant_pre_train']
        self.emb_size = global_params['embedding_size']
        self.latent_space_size = encoder_params['latent_space_size']
        self.projection = nn.Sequential(
            nn.Linear(in_features=self.emb_size * self.magic, out_features=self.emb_size * self.magic), #TODO findt out a way to calculate this constant
            nn.BatchNorm1d(self.emb_size * self.magic),
            nn.ReLU(),
            nn.Linear(in_features=self.emb_size * self.magic, out_features=self.latent_space_size),
            nn.BatchNorm1d(self.latent_space_size),
        )

    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1) # flatten it.
        x = self.projection(x)
        return x

