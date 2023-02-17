"""Following SimCLR CL structure
"""
import torch.nn as nn
import torch.nn.functional as fn
import torch

from modules import ConvolutionalEncoder, Projector, DownstreamClassifier


class SeqCLR(nn.Module):
    def __init__(self):
        self.encoder = ConvolutionalEncoder()
        self.projector = Projector()
        self.classifier = DownstreamClassifier()

    def forward(self, x1, x2):
        # May be faster to parallelize this by keeping an exact copy of the encoder
        x1 = self.encoder(x1)
        x1 = self.projector(x1)

        x2 = self.encoder(x2)
        x2 = self.projector(x2)

        return x1, x2

def contrastive_training():
    batch = []  # Make it a list of signals or something
    for signal in batch:
        s1, s2 =  # Two augmentations

if __name__ == '__main__':
    loader = torch.
