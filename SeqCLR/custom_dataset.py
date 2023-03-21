import os
import random
import typing
import bisect

import mne
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from braindecode import augmentation
from braindecode.datasets.base import BaseConcatDataset
import matplotlib.pyplot as plt

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


class ContrastiveAugmentedDataset(BaseConcatDataset):
    """
    BaseConcatDataset is a ConcatDataset from pytorch, which means thath this should be ok.
    """

    def __init__(self, list_of_ds, target_transform=None, random_state=None):
        super().__init__(list_of_ds, target_transform=target_transform)
        if random_state == None:
            random_state = 100
        # TODO: select correct augmentations and parameters
        self.augmentation_names = ['dropout', 'additive_noise', 'freq_shift']
        self.augmentations = [augmentation.ChannelsDropout,
                              augmentation.GaussianNoise,
                              augmentation.FrequencyShift]
        self.augment_params = [{'p_drop': 0.2, 'random_state': random_state},
                               {'std': 8, 'random_state': random_state},
                               {'sfreq': 250, 'delta_freq': 10}]
        """
        (probability=1, sfreq=180, bandwidth=1, max_freq=None, random_state=random_state)
        (probability=1, p_drop=0.2, random_state=random_state)
        (probability=1, std=0.1, random_state=random_state)
        (probability=1, sfreq=180, max_delta_freq=2, random_state=random_state)
        """

    def __getitem__(self, idx):
        """
        goal is to create pairs of form [x_augment_1, x_augment_2], x_original
        :param idx: idx of the dataset we are interested in
        :return:
        """
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        sample = self.datasets[dataset_idx][sample_idx][0]

        sample = torch.Tensor(sample).view(-1, sample.shape[0], sample.shape[1])
        augmentation_id = random.sample(range(0, len(self.augmentations)), 2)
        # apply augmentations

        aug_1, aug_2 = self.augmentations[augmentation_id[0]], self.augmentations[augmentation_id[1]]
        param_1, param_2 = self.augment_params[augmentation_id[0]], self.augment_params[augmentation_id[1]]

        augmented_1 = aug_1.operation(sample, y=None, **param_1)[0][0]
        augmented_2 = aug_2.operation(sample, y=None, **param_2)[0][0]

        #self.print_channels_and_diff(sample,augmented_1, augmentation_id[0], 0)
        # print(augmented_1.shape, augmented_2.shape, sample.shape)

        return augmented_1,augmented_2, sample[0]

    def get_splits(self, TRAIN_SPLIT:float):
        """
        :param TRAIN_SPLIT: percentage size of train dataset compared to original dataset
        :return: train,test, train and test of instances ContrastiveAugmentedDataset
        """
        split_dict = {'test': range(round(len(self.datasets) * (1 - TRAIN_SPLIT))),
                      'train': range(round(len(self.datasets) * (1 - TRAIN_SPLIT)),
                                     round(len(self.datasets)))}
        splitted = self.split(by=split_dict)
        assert splitted['test'].__len__() + splitted['train'].__len__() == self.__len__()
        assert list(set(splitted['test'].datasets) & set(splitted['train'].datasets)) == []
        train, test = ContrastiveAugmentedDataset(splitted['train'].datasets), ContrastiveAugmentedDataset(
            splitted['test'].datasets)

        return train,test

    def print_channels_and_diff(self, sample, augmented, augmentation_id,channel):
        diff = sample[0][channel].detach().numpy() - augmented[channel].detach().numpy()
        figs, axs = plt.subplots(1, 3)
        axs[0].plot(sample[0][channel].detach().numpy())
        axs[0].set_title('sample')
        axs[1].plot(augmented[channel].detach().numpy())
        axs[1].set_title('augmented ' + self.augmentation_names[augmentation_id])
        axs[2].plot(diff)
        axs[2].set_title('difference')
        plt.show()

    def _get_sequence(self, indices):
        X, y = list(), list()
        for ind in indices:
            out_i = super().__getitem__(ind)
            X.append(out_i[0])
            y.append(out_i[1])

        X = np.stack(X, axis=0)
        y = np.array(y)

        return X, y

    """
    def __getitem__(self, idx):
        
        Parameters
        ----------
        idx : int | list
            Index of window and target to return. If provided as a list of
            ints, multiple windows and targets will be extracted and
            concatenated. The target output can be modified on the
            fly by the ``traget_transform`` parameter.
        
        if isinstance(idx, typing.Iterable):  # Sample multiple windows
            item = self._get_sequence(idx)
        else:
            item = super().__getitem__(idx)
        if self.target_transform is not None:
            item = item[:1] + (self.target_transform(item[1]),) + item[2:]
        return item
    """
