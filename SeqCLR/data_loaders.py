import torch
from torch.utils import data
import numpy as np
import pickle as pkl
import random
from scipy.signal import iirnotch, lfilter

class ContrastiveDataset(data.Dataset):
    def __init__(self, cached_ps_dataset, path=None, total_points=None, window_size=None, sfreq=1000, bw=5, 
                 randomized_augmentation=False, num_channels=11, temporal_len=3000, 
                 windowed_data_name="_Windowed_Pretext_Preprocess.npy", windowed_start_time_name="_Windowed_StartTime.npy"):
        
        if cached_ps_dataset is not None:
            self.init_from_cached_data(cached_ps_dataset)
        else:
            self.init_params_from_scratch(path, 
                                          total_points, 
                                          window_size, 
                                          sfreq, 
                                          bw, 
                                          randomized_augmentation, 
                                          num_channels, 
                                          temporal_len, 
                                          windowed_data_name, 
                                          windowed_start_time_name
            )
        pass 

    def init_params_from_scratch(self, path, total_points, window_size, sfreq, bw, 
                                 randomized_augmentation, num_channels, temporal_len, 
                                 windowed_data_name="_Windowed_Pretext_Preprocess.npy", 
                                 windowed_start_time_name="_Windowed_StartTime.npy"):
        self.available_augmentations = {
            "amplitude_scale": [0.5, 2], 
            "time_shift": [-50, 50], 
            "DC_shift": [-10, 10], 
            "zero-masking": [0, 150], 
            "additive_Gaussian_noise": [0, 0.2], 
            "band-stop_filter": [2.8, 82.5], 
        }
        self.TEMPORAL_DIM = 0
        self.CHANNEL_DIM = 1
        self.NUM_AUGMENTATIONS = 2
        self.NUM_CHANNELS = num_channels
        self.TEMPORAL_LEN = temporal_len
        self.SFREQ = sfreq
        self.BW = bw # band width (?) see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirnotch.html

        data_path = path + windowed_data_name
        self.data = np.load(data_path)
        # print("SQDataset.init_params_from_scratch(): data size == ", self.data.size)
        # print("SQDataset.init_params_from_scratch(): data shape == ", self.data.shape)

        data_path = path + windowed_start_time_name
        self.start_times = np.load(data_path)
        self.total_windows = len(self.data)
        self.randomized_augmentation = randomized_augmentation
        self.pairs, self.labels = self.get_samples_and_labels(size=total_points, window_size=window_size)
        pass
    
    def init_from_cached_data(self, cached_rp_dataset):
        cached_dataset = None
        with open(cached_rp_dataset, 'rb') as infile:
            cached_dataset = pkl.load(infile)
        
        self.available_augmentations = cached_dataset['available_augmentations']
        self.TEMPORAL_DIM = cached_dataset['TEMPORAL_DIM']
        self.CHANNEL_DIM = cached_dataset['CHANNEL_DIM']
        self.NUM_AUGMENTATIONS = cached_dataset['NUM_AUGMENTATIONS']
        self.NUM_CHANNELS = cached_dataset['NUM_CHANNELS']
        self.TEMPORAL_LEN = cached_dataset['TEMPORAL_LEN']
        self.SFREQ = cached_dataset['SFREQ']
        self.BW = cached_dataset['BW']
        self.data = cached_dataset['data']
        # print("SQDataset.init_from_cached_data(): data size == ", self.data.size)
        # print("SQDataset.init_from_cached_data(): data shape == ", self.data.shape)
        self.start_times = cached_dataset['start_times']
        self.total_windows = cached_dataset['total_windows']
        self.randomized_augmentation = cached_dataset['randomized_augmentation']
        self.pairs = cached_dataset['pairs']
        self.labels = cached_dataset['labels']

        del cached_dataset
        pass

    def save_as_dictionary(self, path):
        with open(path, 'wb') as outfile:
            pkl.dump({
                'available_augmentations': self.available_augmentations, 
                'TEMPORAL_DIM': self.TEMPORAL_DIM, 
                'CHANNEL_DIM': self.CHANNEL_DIM, 
                'NUM_AUGMENTATIONS': self.NUM_AUGMENTATIONS, 
                'NUM_CHANNELS': self.NUM_CHANNELS, 
                'TEMPORAL_LEN': self.TEMPORAL_LEN, 
                'SFREQ': self.SFREQ, 
                'BW': self.BW, 
                'data': self.data,
                'start_times': self.start_times,
                'total_windows': self.total_windows,
                'randomized_augmentation': self.randomized_augmentation, 
                'pairs': self.pairs,
                'labels': self.labels,
            }, outfile)

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        x = self.data[self.pairs[index][0], :, :]
        x_aug = self.data[self.pairs[index][0], :, :]
        if self.pairs[index][1] is not None: 
            # print("SQDataset.__getitem__: using STORED augmentations set self.pairs[index][1] == ", self.pairs[index][1])
            x_aug = self.apply_augmentations(x_aug, self.pairs[index][1])
        else:
            # print("SQDataset.__getitem__: using NEW augmentations set because self.pairs[index][1] == ", self.pairs[index][1])
            curr_augmentations = self.get_augmentation_set()
            # print("SQDataset.__getitem__: using NEW augmentations set curr_augmentations == ", curr_augmentations)
            x_aug = self.apply_augmentations(x_aug, curr_augmentations)

        x = torch.from_numpy(x).float()
        x_aug = torch.from_numpy(x_aug).float()
        return x, x_aug

    def get_samples_and_labels(self, size, window_size):
        """
        Gets the pairs of inputs [x, [t1, t2]] and output labels for the pretext task. 
        see Section 2.3 of proceedings.mlr.press/v136/mohsenvand20a/mohsenvand20a.pdf
        """
        pairs = [[None, None] for _ in range(size)] # np.zeros((size, 2), dtype=int)
        labels = None

        for i in range(size):
            anchor_val = np.random.randint(low=0, high=self.total_windows)
            second_val = None
            if not self.randomized_augmentation: # decide whether or not to save augmentation strategy for curr sample
                second_val = self.get_augmentation_set()
            # else:
            #     second_val = None

            pairs[i][0] = anchor_val  # type: ignore
            pairs[i][1] = second_val  # type: ignore

        # print("PSDataset.get_samples_and_labels: labels shape == ", labels.shape)
        return pairs, labels

    def get_augmentation_set(self):
        """
        see Section 2.3 of proceedings.mlr.press/v136/mohsenvand20a/mohsenvand20a.pdf
        """
        augmentation_set = [] # [dict()]*self.NUM_CHANNELS
        
        for j in range(self.NUM_CHANNELS):
            augmentation_set.append(dict())
            selected_augmentations = random.sample(list(self.available_augmentations.keys()), self.NUM_AUGMENTATIONS)
            assert len(selected_augmentations) == 2
            # print("selected_augmentations == ", selected_augmentations)
            # print("SQDataset.get_augmentation_set: len(selected_augmentations) == 2")
            counter = 0
            for _, curr_augmentation in enumerate(selected_augmentations):
                curr_augmentation_val = None

                if curr_augmentation in ['amplitude_scale', 'DC_shift', 'additive_Gaussian_noise', 'band-stop_filter']: # augmentation that requires float val
                    curr_augmentation_val = random.uniform(self.available_augmentations[curr_augmentation][0], self.available_augmentations[curr_augmentation][1]) # see https://stackoverflow.com/questions/6088077/how-to-get-a-random-number-between-a-float-range

                elif curr_augmentation in ['time_shift', 'zero-masking']: # augmentation that requires int val
                    curr_augmentation_val = random.randint(self.available_augmentations[curr_augmentation][0], self.available_augmentations[curr_augmentation][1]) # see https://stackoverflow.com/questions/3996904/generate-random-integers-between-0-and-9
                    if curr_augmentation == 'zero-masking':
                        curr_augmentation_val = [curr_augmentation_val, random.randint(0, self.TEMPORAL_LEN-1)]

                else:
                    raise NotImplementedError("curr_augmentation == "+str(curr_augmentation)+" not recognized for value sampling")

                augmentation_set[j][curr_augmentation] = curr_augmentation_val
                counter += 1
            # print("augmentation_set == ", augmentation_set)
            # print("augmentation_set[j].keys() == ", augmentation_set[j].keys())
            assert len(list(augmentation_set[j].keys())) == 2
            assert counter == 2
        # raise NotImplementedError()
        return augmentation_set

    def apply_augmentations(self, x, augmentations):
        """
        see Section 2.2 of proceedings.mlr.press/v136/mohsenvand20a/mohsenvand20a.pdf
        """
        assert len(augmentations) == 11

        for j, curr_augmentation_set in enumerate(augmentations):
            assert len(list(curr_augmentation_set.keys())) == 2

            for _, curr_augmentation in enumerate(list(curr_augmentation_set.keys())):
                
                curr_augmentation_val = curr_augmentation_set[curr_augmentation]

                if curr_augmentation == 'amplitude_scale':
                    x[:,j] = curr_augmentation_val * x[:,j]
                    
                elif curr_augmentation == 'DC_shift':
                    x[:,j] = x[:,j] + curr_augmentation_val
                    
                elif curr_augmentation == 'additive_Gaussian_noise':
                    # see https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python
                    # and https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
                    x[:,j] = x[:,j] + np.random.normal(0, curr_augmentation_val, x[:,j].shape)
                    
                elif curr_augmentation == 'band-stop_filter':
                    # see:
                    #     https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirnotch.html
                    #     https://www.programcreek.com/python/example/115815/scipy.signal.iirnotch
                    #     https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
                    b, a = iirnotch(curr_augmentation_val, curr_augmentation_val/self.BW, self.SFREQ)
                    x[:,j] = lfilter(b, a, x[:,j])

                elif curr_augmentation == 'time_shift':
                    if curr_augmentation_val != 0:
                        new_signal = np.zeros(x[:,j].shape)
                        
                        if curr_augmentation_val < 0:
                            new_signal[:curr_augmentation_val] = x[np.abs(curr_augmentation_val):,j]
                            new_signal[curr_augmentation_val:] = x[:np.abs(curr_augmentation_val),j]
                        else:
                            new_signal[:curr_augmentation_val] = x[-curr_augmentation_val:,j]
                            new_signal[curr_augmentation_val:] = x[:-curr_augmentation_val,j]
                            
                        x[:,j] = new_signal

                elif curr_augmentation == 'zero-masking':
                    x[curr_augmentation_val[1]:curr_augmentation_val[1]+curr_augmentation_val[0], j] = 0.
                
                else:
                    raise NotImplementedError("curr_augmentation == "+str(curr_augmentation)+" not recognized for application")
        
        # print("SQDataset.apply_augmentations: x shape == ", x.shape)
        return x