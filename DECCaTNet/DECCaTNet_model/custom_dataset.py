import json
import os
import random
# import typing
import bisect
from random import sample
import mne
import torch
# import pandas as pd
# from skimage import io, transform
# import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from braindecode import augmentation
from braindecode.datasets.base import BaseConcatDataset
import matplotlib.pyplot as plt
from braindecode.datautil.serialization import _load_parallel, _load_signals
# from braindecode.datautil.serialization import load_concat_dataset
from DECCaTNet_model.augmentations import SignalPermutation, Scale, TimeShift, AddNoise
# Ignore warnings
import warnings
import torchplot as plt

warnings.filterwarnings("ignore")


class ConcatPathDataset(ConcatDataset):
    """
    ConcatDataset of different PathDatasets, containing several PathDatasets. This class will allow us to sample
    randomly from different PathDatasets
    """

    def __init__(self, dataset_dict: dict, all_params, global_params, splitted_datasets=None, dataset_names=None):
        """
        :param dataset_dict: dict with format key: (path,ids), used to init PathDataset
        :param random_state:
        :param splitted_datasets: Only used when splitting dataset, dont need to initialize new PathDatasets
        """
        self.all_params = all_params
        self.global_params = global_params
        self.dataset_names = []
        if splitted_datasets is None:
            datasets = []
            for dataset in dataset_dict.keys():
                path_dataset = PathDataset(ids_to_load=dataset_dict[dataset][1], path=dataset_dict[dataset][0],
                                           all_params=self.all_params, global_params=self.global_params)
                datasets.append(path_dataset)
                self.dataset_names.append(dataset)
        else:
            self.dataset_names = dataset_names
            datasets = splitted_datasets
        super().__init__(datasets)

    def get_splits(self, TRAIN_SPLIT: float):
        """
        :param TRAIN_SPLIT: percentage size of train dataset compared to original dataset
        :return: train,test, intances of ConcatPathDatasets with the correct training and testing ids
        """
        train_ds = []  # init list of ds for training and testing
        test_ds = []
        for dataset in self.datasets:  # iterate through datasets
            train, test = dataset.get_splits(TRAIN_SPLIT)  # use datasets get splits function, return two pathDatasets
            train_ds.append(train)
            test_ds.append(test)

        return ConcatPathDataset(None, self.all_params,self.global_params,
                                 splitted_datasets=train_ds, dataset_names=self.dataset_names), \
               ConcatPathDataset(None, self.all_params,self.global_params,
                                 splitted_datasets=test_ds,
                                 dataset_names=self.dataset_names)  # return two new ConcatPathDatasets


def select_params(param: dict):
    for key in param.keys():
        value = param[key]
        if isinstance(value, tuple):
            if isinstance(value[0], float) or isinstance(value[1], float):
                value_1 = random.randint(value[0] * 10, value[
                    1] * 10) / 10  # some values are float with one decimal, but can go around this by multiplying by 10
            elif isinstance(value[0], torch.Tensor):
                if isinstance(value[0][0].item(), float):
                    value_1 = torch.Tensor(
                        [random.randint(round(value[0][0].item() * 10), round(value[1][0].item() * 10)) / 10])
                else:
                    value_1 = torch.Tensor([random.randint(value[0][0].item(), value[1][0].item())])
            else:
                value_1 = random.randint(value[0], value[1])
            param[key] = value_1
    # print(f'param for denne kjøringen er {param}')
    return param


class PathDataset(Dataset):
    """
    BaseConcatDataset is a ConcatDataset from pytorch, which means thath this should be ok.
    """

    def __init__(self, ids_to_load, path, all_params, global_params):
        self.dataset_names = all_params['preprocess']['source_ds']
        self.all_params = all_params
        self.global_params = global_params

        self.ids_to_load = ids_to_load
        self.path = path

        self.noise_probability = self.all_params['pre_training']['augmentation']['noise_probability']
        self.sfreq = self.global_params['s_freq']
        self.random_state = self.global_params['random_state']

        self.preload = False
        self.is_raw = False

        # random.seed(self.random_state)
        # legal combinations of augmentations: All of the augmentations below are "Weak"

        # TODO: select correct parameters and write

        self.augmentation_names = ['permutation', 'masking', 'bandstop', 'gaussian', 'freq_shift', 'scale',
                                   'time_shift']
        self.augmentations = {'permutation': SignalPermutation,
                              'masking': augmentation.SmoothTimeMask,
                              'bandstop': augmentation.BandstopFilter,
                              'gaussian': augmentation.GaussianNoise,
                              'freq_shift': augmentation.FrequencyShift,
                              'scale': Scale,
                              'time_shift': TimeShift,
                              'add_noise': AddNoise,

                              }
        self.augment_params = {'permutation': {'n_permutations': (5, 10)},
                               'masking': {'mask_start_per_sample': (torch.Tensor([1000]), torch.Tensor([2000])),
                                           # TODO we have different sequence length, so this should be changed
                                           'mask_len_samples': (150, 1000)},
                               'bandstop': {'sfreq': self.sfreq, 'bandwidth': 5,  # TODO same here, bandwidth length
                                            'freqs_to_notch': (torch.Tensor([20.]), torch.Tensor([82.5]))},
                               'gaussian': {'std': (0, 0.2)},
                               'freq_shift': {'delta_freq': (-10, 10), 'sfreq': self.sfreq},
                               'scale': {'scale_factor': (0.5, 2)},
                               'time_shift': {'time_shift': (-50, 50)},  # TODO same here
                               'add_noise': {'std': (0, 0.05)}

                               # {'p_drop': 0.2, 'random_state': random_state},
                               #                    {'std': 8, 'random_state': random_state},
                               # {'sfreq': 250, 'delta_freq': 10}
                               }
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

        signals = _load_signals(fif_file_path, self.preload, self.is_raw)
        sample = signals.get_data(item=window_n)
        sample = torch.Tensor(sample)

        if sample.shape[2] != 9600: # TODO fix this in preprocessing
            # print(sample.shape,fif_file_name, window_n)
            sample = torch.nn.functional.pad(input=sample,pad=(0,9600-sample.shape[-1],0,0,0,0),mode='constant',value=0)

        # apply augmentations
        augmentation_id = random.sample(range(0, len(self.augmentation_names)), 2)
        aug_1 = self.augmentation_names[augmentation_id[0]]
        aug_2 = self.augmentation_names[augmentation_id[1]]
        param_1 = self.augment_params[aug_1].copy()
        param_2 = self.augment_params[aug_2].copy()

        param_1 = select_params(param_1)
        param_2 = select_params(param_2)

        if self.noise_probability > random.random() and aug_1 != 'gaussian':
            param_noise = select_params(self.augment_params['add_noise'].copy())
            augmented_1 = \
                self.augmentations['add_noise'].operation(sample, None, **param_noise, aug=self.augmentations[aug_1],
                                                          params=param_1)[0]
        else:
            augmented_1 = self.augmentations[aug_1].operation(sample, y=None, **param_1)[0]

        if self.noise_probability > random.random() and aug_2 != 'gaussian':
            param_noise = select_params(self.augment_params['add_noise'].copy())
            augmented_2 = \
                self.augmentations['add_noise'].operation(sample, None, **param_noise, aug=self.augmentations[aug_2],
                                                          params=param_2)[0]
        else:
            augmented_2 = self.augmentations[aug_2].operation(sample, y=None, **param_2)[0]

        # self.visualize_augmentations(sample, augmented_1, augmented_2, augmentation_id, plot_diff=True)

        return augmented_1, augmented_2, sample

    def get_splits(self, TRAIN_SPLIT: float):
        """
        :param TRAIN_SPLIT: percentage size of train dataset compared to original dataset
        :return: train,test, train and test of instances PathDataset
        """
        train_ids = sample(self.ids_to_load, round(len(self.ids_to_load) * TRAIN_SPLIT))
        test_ids = list(set(self.ids_to_load) - set(train_ids))
        assert len(train_ids) + len(test_ids) == len(self.ids_to_load)

        return PathDataset(train_ids, path=self.path, all_params=self.all_params, global_params=self.global_params), \
               PathDataset(test_ids, path=self.path, all_params=self.all_params, global_params=self.global_params)

    def visualize_augmentations(self, original, augmented_1, augmented_2, augmentation_id, plot_diff=False):
        """
        Visualize the effect of augmentations on the original signal
        :param plot_diff: if differences should be plotted or not.
        :param original: original signal which has been augmented
        :param augmented_1: augmented signal with augmentation 1
        :param augmented_2: augmented signal with augmentation 2
        :param augmentation_id: ids of the two selected augmentations
        :return: a plot of the different augmentations
        """
        # get n_channels
        n_channels = original.shape[1]

        # reshape all for easier plotting
        original = original[0]
        augmented_1 = augmented_1[0]
        augmented_2 = augmented_2[0]

        # create on plot for each channel
        if plot_diff:
            dims = n_channels + 1
        else:
            dims = n_channels
        figs, axs = plt.subplots(dims, 2)
        for i in range(n_channels):
            axs[i, 0].plot(original[i], color='orange', label='original')
            axs[i, 1].plot(original[i], color='orange', label='original')
            axs[i, 0].plot(augmented_1[i], color='blue', alpha=0.5, label=self.augmentation_names[augmentation_id[0]])
            axs[i, 1].plot(augmented_2[i], color='blue', alpha=0.5, label=self.augmentation_names[augmentation_id[1]])
            axs[i, 0].axis('off')
            axs[i, 1].axis('off')
            axs[i, 0].legend(loc='upper left')
            axs[i, 1].legend(loc='upper left')
            if plot_diff:
                axs[-1, 0].plot(original[i] - augmented_1[i], alpha=0.5, label=f'diff for channel {i}')
                axs[-1, 1].plot(original[i] - augmented_2[i], alpha=0.5, label=f'diff for channel {i}')
                axs[-1, 0].legend(loc='upper left')
                axs[-1, 1].legend(loc='upper left')
                axs[-1, 0].axis('off')
                axs[-1, 1].axis('off')

        axs[0, 1].title.set_text(self.augmentation_names[augmentation_id[1]])
        axs[0, 0].title.set_text(self.augmentation_names[augmentation_id[0]])
        plt.tight_layout(pad=0.3, w_pad=0.5, h_pad=0.5)
        plt.show()


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

        augmented_1 = aug_1.operation(sample, y=None, **param_1)[0]
        augmented_2 = aug_2.operation(sample, y=None, **param_2)[0]

        # self.print_channels_and_diff(sample,augmented_1, augmentation_id[0], 0)
        # print(augmented_1.shape, augmented_2.shape, sample.shape)

        return augmented_1, augmented_2, sample

    def get_splits(self, TRAIN_SPLIT: float):
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

        return train, test

    def print_channels_and_diff(self, sample, augmented, augmentation_id, channel):
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
