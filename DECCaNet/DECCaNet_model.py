import torch
import torch.nn as nn


class DECCaNet(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        emb_size = 32
        latent_space_size = 64
        self.encoder = Encoder(emb_size, batch_size)
        self.projector = Projector(emb_size,latent_space_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.projector(x)
        return x


class Encoder(nn.Module):
    def __init__(self,emb_size,batch_size):
        super().__init__()
        self.ConvEmbedding = Convolution(emb_size,batch_size)
        self.TransEncoder = TransEncoder(emb_size)

    def forward(self, x):
        x = self.ConvEmbedding(x)
        return self.TransEncoder(x)


class Convolution(nn.Module):
    """
    A traditional Transformer uses potitionalembedding to capture local features
    However we will use a CNN instead. This is due to the discovered effects
    in litterature in combining CNN's and transformers for EEG classification
    """

    def __init__(self, emb_size, batch_size):
        super().__init__()
        temporal = 40
        spatial = 40
        dropout = 0.5
        self.emb_size = emb_size
        self.n_channels = 1
        self.n_samples = 15000
        self.batch_size = batch_size
        self.temporal = nn.Sequential(nn.Conv2d(self.n_channels, temporal, kernel_size=(1, 25), stride=(1, 1))
                                      )
        self.spatial = nn.Sequential(nn.Conv2d(temporal, spatial, kernel_size=(2, 1), stride=(1, 15)))
        self.pooling = nn.Sequential(nn.BatchNorm2d(spatial),
                                     nn.ELU(),
                                     nn.AvgPool2d((1,75), (1,15))
                                     )
        self.dropout = nn.Dropout(dropout)

        self.projector = nn.Sequential(nn.Conv2d(40, self.emb_size, kernel_size=(1,1), stride=(1,1)))

        # self.spatial = nn.Sequential(
        #
        # )
        # self.projector = nn.Sequential(
        #
        # )

    def forward(self, x):
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.pooling(x)
        x = self.dropout(x)
        x = self.projector(x)
        # has shape: [Batch,embedding_size,1,62] little bit weird, but ok
        x = x.view(self.batch_size,-1, self.emb_size)
        return x


class TransEncoder(nn.Module):
    def __init__(self,emb_size):
        super().__init__()
        self.emb_size = emb_size
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.emb_size, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=10)

    def forward(self, x):
        x = self.encoder(x)
        return x


class Projector(nn.Module):
    def __init__(self, emb_size, latent_space_size):
        super().__init__()
        self.emb_size = emb_size
        self.latent_space_size = latent_space_size
        self.projection = nn.Sequential(
            nn.Linear(in_features=self.emb_size*62, out_features=self.emb_size*62),
            nn.BatchNorm1d(self.emb_size*62),
            nn.ReLU(),
            nn.Linear(in_features=self.emb_size*62, out_features=latent_space_size),
            nn.BatchNorm1d(latent_space_size),
        )


    def forward(self, x:torch.Tensor):
        x = x.view(x.size(0),-1)
        x = self.projection(x)
        return x


class ClassificationHead(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        pass
