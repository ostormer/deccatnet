import json
import os
import random
import typing
import bisect
from random import sample
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
from braindecode.datautil.serialization import _load_parallel, _load_signals
from braindecode.datautil.serialization import load_concat_dataset
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

    #

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
        sample = self.datasets[dataset_idx][sample_idx]
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
        # TODO: Visualize how the different augmentations work
        diff = sample[0][channel].detach().numpy() - augmented[channel].detach().numpy()
        figs, axs = plt.subplots(1, 3)
        axs[0].plot(sample[0][channel].detach().numpy())
        axs[0].set_title('sample')
        axs[1].plot(augmented[channel].detach().numpy())
        axs[1].set_title('augmented ' + self.augmentation_names[augmentation_id])
        axs[2].plot(diff)
        axs[2].set_title('difference')
        plt.show()

class PathDataset(Dataset):
    """
    BaseConcatDataset is a ConcatDataset from pytorch, which means thath this should be ok.
    """

    def __init__(self, ids_to_load, path, preload=False,random_state=None, SSL=True):

        self.ids_to_load = ids_to_load
        self.path = path
        self.preload = preload
        self.is_raw = False
        self.SSL = SSL

        if random_state == None:
            self.random_state = 100
        else:
            self.random_state = random_state
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

    def __len__(self):
        return len(self.ids_to_load)

    def __getitem__(self, idx):
        """
        goal is to create pairs of form [x_augment_1, x_augment_2], x_original
        :param idx: idx of the dataset we are interested in
        :return:
        """
        # form of self.
        mne.set_log_level('ERROR')
        path_i, window_n = self.ids_to_load[idx]

        sub_dir = os.path.join(self.path, str(path_i))
        file_name_patterns = ['{}-raw.fif', '{}-epo.fif']
        fif_name_pattern = file_name_patterns[0] if self.is_raw else file_name_patterns[1]
        fif_file_name = fif_name_pattern.format(path_i)
        fif_file_path = os.path.join(sub_dir, fif_file_name)

        if not self.SSL:
            #TODO: keep target when reading a non SSL dataset
            target_file_path = os.path.join(sub_dir, 'target_name.json')
            target = json.load(open(target_file_path, "r"))['pathological']

        signals = _load_signals(fif_file_path, self.preload, self.is_raw)
        sample = signals.get_data(item=window_n)
        sample = torch.Tensor(sample)

        # apply augmentations
        augmentation_id = random.sample(range(0, len(self.augmentations)), 2)
        aug_1, aug_2 = self.augmentations[augmentation_id[0]], self.augmentations[augmentation_id[1]]
        param_1, param_2 = self.augment_params[augmentation_id[0]], self.augment_params[augmentation_id[1]]

        augmented_1 = aug_1.operation(sample, y=None, **param_1)[0][0]
        augmented_2 = aug_2.operation(sample, y=None, **param_2)[0][0]

        return augmented_1,augmented_2, sample[0]

    def get_splits(self, TRAIN_SPLIT:float):
        """
        :param TRAIN_SPLIT: percentage size of train dataset compared to original dataset
        :return: train,test, train and test of instances PathDataset
        """
        train_ids = sample(self.ids_to_load, round(len(self.ids_to_load)*TRAIN_SPLIT))
        test_ids = list(set(self.ids_to_load)-set(train_ids))
        assert len(train_ids) + len(test_ids) == len(self.ids_to_load)
        return PathDataset(train_ids,path=self.path,preload=self.preload, random_state=self.random_state),\
               PathDataset(test_ids,path=self.path,preload=self.preload, random_state=self.random_state)

    def print_channels_and_diff(self, sample, augmented, augmentation_id,channel):
        # TODO: Visualize how the different augmentations work
        diff = sample[0][channel].detach().numpy() - augmented[channel].detach().numpy()
        figs, axs = plt.subplots(1, 3)
        axs[0].plot(sample[0][channel].detach().numpy())
        axs[0].set_title('sample')
        axs[1].plot(augmented[channel].detach().numpy())
        axs[1].set_title('augmented ' + self.augmentation_names[augmentation_id])
        axs[2].plot(diff)
        axs[2].set_title('difference')
        plt.show()
