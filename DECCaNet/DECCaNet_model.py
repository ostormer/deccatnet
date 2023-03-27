import torch.nn as nn


class DECCaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        # self.projector = Projector()

    def forward(self, x):
        return self.encoder(x)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ConvEmbedding = Convolution()
        self.TransEncoder = TransEncoder()

    def forward(self, x):
        x = self.ConvEmbedding(x)
        return self.TransEncoder(x)


class Convolution(nn.Module):
    """
    A traditional Transformer uses potitionalembedding to capture local features
    However we will use a CNN instead. This is due to the discovered effects
    in litterature in combining CNN's and transformers for EEG classification
    """

    def __init__(self):
        super().__init__()
        temporal = 40
        spatial = 40
        dropout = 0.5
        emb_size = 32
        self.n_channels = 1
        self.n_samples = 15000
        self.batch_size = 10
        self.temporal = nn.Sequential(nn.Conv2d(self.n_channels, temporal, kernel_size=(1, 25), stride=(1, 1))
                                      )
        self.spatial = nn.Sequential(nn.Conv2d(temporal, spatial, kernel_size=(2, 1), stride=(1, 15)))
        self.pooling = nn.Sequential(nn.BatchNorm2d(spatial),
                                     nn.ELU(),
                                     nn.AvgPool2d((1,75), (1,15))
                                     )
        self.dropout = nn.Dropout(dropout)

        self.projector = nn.Sequential(nn.Conv2d(40, emb_size, kernel_size=(1,1), stride=(1,1)))

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
        return x


class TransEncoder(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, X):
        pass


class Projector(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        pass


class ClassificationHead(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        pass
