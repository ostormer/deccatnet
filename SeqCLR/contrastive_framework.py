"""Following SimCLR CL structure
"""
import torch.nn as nn
import torch.nn.functional as fn
import torch
import numpy as np

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

# start by creating contrastive loss function, first is inspired by (Section 2.3 of proceedings.mlr.press/v136/mohsenvand20a/mohsenvand20a.pdf)
# which is a temperature contrastive loss

class ContrastiveLoss(nn.Module):
    """
    ContrastiveLoss module, can be used as a normal nn.Module as it has a forward function. The forward function takes
        two sets as input and returns a loss. This implementation is inspired by SimCLR Chen et al(2020) and SeqCLR Moshenvand
        et al (2021).
    """
    def __init__(self,temperature):
        """
        Initiializes the Contrastive loss module
        :param temperature: Learnable temperature parameter
        """
        super(ContrastiveLoss,self).__init__() #call super to access and overwrite nn.Module methods
        self.tau = temperature
        self.BATCH_DIM = 0 # the dimension in z.size which has the batch size
        self.cos_sim = nn.CosineSimilarity(0) # use cosine similiarity as similairity measurement

    def forward(self, z1:torch.Tensor, z2:torch.Tensor):
        """
        Called whenever ContrastiveLoss class is called after initialization
        In original paper, only z2 is assumed to be augmented, we augment both input signals
        :param z1,z2: Latent space representations of pairwise positive pairs as numpy arrays, same indexes should
            be positive pairs
        :return: ContrastiveLoss maximizing agreement between pairs and minimizing agreement with
            negative pairs
        """
        contrastive_loss = 0. # set loss
        batch_size = z1.size(self.BATCH_DIM) # get batch size from input

        # normalize data in batch as is has increased performance, however may be double
        # TODO: check if double normalization
        z1 = fn.normalize(z1, dim=1)
        z2 = fn.normalize(z2, dim=1)

        # TODO: check z1 before and after view, write comment
        z1 = z1.view(batch_size, -1)
        z2 = z2.view(batch_size, -1)

        for i in range(batch_size):
            # iterate over the entire batch, calculate loss with regard to one and one sample

            # calculte loss contirbutions from set z1 on z2
            cosine_pair = self.cos_sim(z1[i,:], z2[i,:]) # compute cosine similarity for positive pair
            num_1 = torch.exp(cosine_pair/self.tau) # get numerator for loss function
            denom_1 = 0. # define denominator in loss function
            for k in range(batch_size):
                # iterate over all samples with regard to sample i, calculate contirbution to loss from each sample
                cosine_z2 = self.cos_sim(z1[i,:], z2[k,:])
                denom_1 += torch.exp(cosine_z2/self.tau) # update denominator, include i == k for because z1 != z2
                if k != i:
                    # avoid calculating loss with regard to itself
                    cosine_z1 = self.cos_sim(z1[i,:], z1[k,:])
                    denom_1 += torch.exp(cosine_z1/self.tau) # update denominator with negative pairs from own set
            contrastive_loss += -1*torch.log(num_1/denom_1) # update loss

            # calculate loss contirbutions from set z2 on z1
            num_2 = num_1 # as CosineSimilarity in pytorch is a cumulative function
            denom_2 = 0. # define new demnominator
            for k in range(batch_size):
                # switch z1 and z2
                cosine_z1 = self.cos_sim(z2[i,:],z1[k,:])
                denom_2 += torch.exp(cosine_z1/self.tau)
                if k  != i:
                    cosine_z2 = self.cos_sim(z2[i,:], z2[k,:])
                    denom_2 += torch.exp(cosine_z2/self.tau)
            contrastive_loss += -1*torch.log(num_2/denom_2)
        # avreage out loss for both batches, minus 1 because cosine_similiarty with itself is not included
        return contrastive_loss/(batch_size*2.*(2.*batch_size - 1.))

class ContrastiveLossGPT(nn.Module):
    """
    ContrastiveLoss module, can be used as a normal nn.Module as it has a forward function. The forward function takes
        two sets as input and returns a loss. This implementation is inspired by SimCLR Chen et al(2020) and SeqCLR Moshenvand
        et al (2021).
    """
    def __init__(self, temperature):
        """
        Initiializes the Contrastive loss module
        :param temperature: Learnable temperature parameter
        """
        super(ContrastiveLoss, self).__init__()
        self.tau = temperature
        self.BATCH_DIM = 0 # the dimension in z.size which has the batch size
        self.cos_sim = nn.CosineSimilarity(0) # use cosine similarity as similarity measurement

    def forward(self, z1:torch.Tensor, z2:torch.Tensor):
        """
        Called whenever ContrastiveLoss class is called after initialization
        In original paper, only z2 is assumed to be augmented, we augment both input signals
        :param z1,z2: Latent space representations of pairwise positive pairs as numpy arrays, same indexes should
            be positive pairs
        :return: ContrastiveLoss maximizing agreement between pairs and minimizing agreement with
            negative pairs
        """
        batch_size = z1.size(self.BATCH_DIM) # get batch size from input

        z1 = fn.normalize(z1, dim=1)
        z2 = fn.normalize(z2, dim=1)

        # compute cosine similarities
        similarities = self.cos_sim(z1, z2) / self.tau

        # create mask to exclude positive pairs from the denominator
        mask = torch.eye(batch_size, device=z1.device)

        # compute numerator and denominator for z1 -> z2 loss
        numerator_1 = torch.exp(similarities) * (1 - mask)
        denominator_1 = torch.exp(similarities) + torch.sum(torch.exp(self.cos_sim(z1, z1) / self.tau), dim=1) - 1

        # compute numerator and denominator for z2 -> z1 loss
        numerator_2 = torch.exp(similarities.t()) * (1 - mask)
        denominator_2 = torch.exp(similarities.t()) + torch.sum(torch.exp(self.cos_sim(z2, z2) / self.tau), dim=1) - 1

        # compute losses
        loss_1 = -torch.sum(torch.log(numerator_1 / denominator_1))
        loss_2 = -torch.sum(torch.log(numerator_2 / denominator_2))

        # average over batch
        loss = (loss_1 + loss_2) / (2.0 * batch_size)

        return loss







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
