from contrastive_framework import pre_train_model
from load_windowed import SingleChannelDataset
import pickle
from tqdm import tqdm
import os
from braindecode.datasets import BaseConcatDataset
import braindecode.datasets.tuh as tuh
from braindecode.preprocessing import create_fixed_length_windows
from torch.utils.data import DataLoader
from mne import set_log_level
import torch


if __name__=="__main__":
    # the goal for this iteration is to run pre_train_model function
    # still need a way to load the dataset,
    READ_CACHED_DS = True  # Change to read cache or not
    SOURCE_DS = 'tuh_eeg'  # Which dataset to load

    assert SOURCE_DS in ['tuh_eeg_abnormal', 'tuh_eeg']
    # Disable most MNE logging output which slows execution
    set_log_level(verbose='WARNING')

    dataset_root = None
    cache_path = None
    dataset = None
    if SOURCE_DS == 'tuh_eeg_abnormal':
        dataset_root = '../datasets/TUH/tuh_eeg_abnormal'
        # remeber to choose correct cache part for your computer, or drive, in addition tuh.py does not work for mye
        # for tuh_eeg_abnormal
        cache_path = '../datasets/tuh_braindecode/styrk_tuh_abnormal.pkl'

    else:
        dataset_root = '../datasets/TUH/tuh_eeg'
        cache_path = '../datasets/tuh_braindecode/styrk_tuh_eeg.pkl'

    # since the pickel files are references to locations on a disk, traversing between computers is hard. However
    # It is possible to do this with

    if READ_CACHED_DS:
        with open(cache_path, 'rb') as f:
            dataset = pickle.load(f)
    else:
        dataset = SingleChannelDataset(dataset_root, SOURCE_DS)

        with open(cache_path, 'wb') as f:
            pickle.dump(dataset, f)

    # print(ds.description)

    subset = dataset.split(by=range(10))['0']
    #print(subset.description)

    subset_windows = create_fixed_length_windows(
        subset,
        picks="eeg",
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=2500,
        window_stride_samples=2500,
        drop_last_window=False,
        # mapping={'M': 0, 'F': 1},  # map non-digit targets
    )
    # store the number of windows required for loading later on
    subset_windows.set_description({
        "n_windows": [len(d) for d in subset_windows.datasets]})  # type: ignore

    # Default DataLoader object lets us iterate through the dataset.
    # Each call to get item returns a batch of samples,
    # each batch has shape: (batch_size, n_channels, n_time_points)
    dl = DataLoader(dataset=subset_windows, batch_size=5)

    batch_X, batch_y, batch_ind = None, None, None
    for batch_X, batch_y, batch_ind in dl:
        pass
    print(batch_X.shape)  # type: ignore
   # print('batch_X:', batch_X)
    print('batch_y:', batch_y)
    print('batch_ind:', batch_ind)

    pre_train_model(batch_size=100, num_workers=1,save_freq=10,Shuffel=False,save_dir_model='models',model_file_name='test',model_weights_dict=None,temperature= 2
                    ,learning_rate= 0.01
                    , weight_decay= 0.01,max_epochs=20,batch_print_condition=5)