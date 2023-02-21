"""Following SimCLR CL structure
"""
import torch.nn as nn
import torch.nn.functional as fn
import torch

from modules import ConvolutionalEncoder, Projector, DownstreamClassifier


"""
SeqCLR contrastive pre-training algortihm summary
Randomly transform a mininbatch of N channels into 2N augmented channels (two augmentations for each channel, creating pairs)

Channel Encoder: learned representations used for downstream tasks
- transform input into latent space representations, in this case it has the same length as the input, but four channels.
    this property is important due to it allowing several different length downstream tasks. 
- two different encoder architectures

Projector: Collapses output of encoder into a 32-dimensional point, this output is where the agreement is maximized

Contrastive loss: (normalized temperature-scaled cross entropy)
Have two outputs from the projector: z_1 and z_2 after sending in augmented views of x. 

In ssl_baselines_zac there is implemented on training algorithm for each model. We will also start off by doing this,
    maybe we will need to switch later

There are limited contrastive modules in the framework, but there is one example, with the contrastive adverserial loss


Overall solution from SSL_baselines_biosignals:

The model is called SeqCLR, however the overall implementation with pre-training etc is called SQnet, so it is a little
    bit tricky to follow the flow in the implementation.
augmentations are applied in the dataset by the overwriting _get_items_ function. Here, also one datapoint/sample is
    selected each time, meaning that this is where one and one channel is selected.
The datset is then loaded into a pytroch dataloader, where a for loop over the dataloader gives data for the pre-training
Assume that all these things is in place, when writing the contrastive framework. Will try to docuemtn well.

Overall flow:
load_dataset -> SQ dataset -> SSL dataset -> pytorch datalaoder -> into for loop in training function for SQnet
SeqCLR modules -> SeqCLR overallclass -> SQNet for pretraing -> pretrained as SQNet

This flow it the reason for no SeqCLR dataset or dataloader etc.
"""

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
