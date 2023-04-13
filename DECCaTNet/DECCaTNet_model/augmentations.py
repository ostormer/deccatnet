import random

import torch
from braindecode import augmentation


def signal_permutate(x, y, n_permutations, random_state=None):
    assert isinstance(n_permutations, (int, torch.IntTensor, torch.cuda.IntTensor)) and n_permutations > 1, (
        f'n_permutations must be an int and larger than 1. Got{n_permutations}.')
    # get number of channels
    n_channels = x.shape[1]
    x = x.view(n_channels, -1)

    sub_tensors = list(torch.tensor_split(x, n_permutations, dim=1))
    random.shuffle(sub_tensors)
    x = torch.cat(sub_tensors,dim=1)
    return x.view(1,n_channels,-1),y

def scale(x,y,scale_factor):
    # TODO: scale must work best if the signal is normalized arounf zero?
    return torch.mul(x,scale_factor),y


def time_shift(x,y,time_shift):
    n_channels = x.shape[1]
    x = x.view(n_channels,-1)
    time_shift = ((time_shift,)*n_channels)
    dims = ((1,)*n_channels)
    return torch.roll(x,time_shift, dims=dims).view(1,n_channels,-1),y


class SignalPermutation(augmentation.Transform):
    operation = staticmethod(signal_permutate)

    def __init__(self, probability, n_permutations, random_state=None):
        super().__init__(probability=probability, random_state=random_state)
        self.n_permutations = n_permutations

    def get_augmentation_params(self, *batch):
        return {
            'n_permutations': self.n_permutations,
            'random_state': self.rng
        }


class Scale(augmentation.Transform):
    operation = staticmethod(scale)

    def __init__(self, probability, scale_factor, random_state=None):
        super().__init__(probability=probability, random_state=random_state)
        self.scale_factor = scale_factor

    def get_augmentation_params(self, *batch):
        return {
            'scale_factor':self.scale_factor
        }


class TimeShift(augmentation.Transform):
    operation = staticmethod(time_shift)

    def __init__(self, probability, time_shift, random_state=None):
        super().__init__(probability=probability, random_state=random_state)
        self.time_shift = time_shift

    def get_augmentation_params(self, *batch):
        return {
            'time_shift':self.time_shift
        }
