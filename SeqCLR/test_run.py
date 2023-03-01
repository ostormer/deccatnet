from contrastive_framework import pre_train_model
from load_windowed import SingleChannelDataset,TUHSingleChannelDataset
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
    READ_CACHED_DS = False # Change to read cache or not
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
        print('creating TUHSingleChannelDataset')
        dataset = TUHSingleChannelDataset(path=dataset_root, source_dataset=SOURCE_DS)
        with open(cache_path, 'wb') as f:
            pickle.dump(dataset, f)

    #print(ds.description)
    #subset = dataset.split(by=range(10))['0']
    """
    NB!! important to not create a subset as this creates a new class which has a new
    __getitem__ function which overrides the original. 
    """
    dl = DataLoader(dataset=dataset, batch_size=5)

    batch_X, batch_y, batch_ind = None, None, None
    for batch_X, batch_y, batch_ind in dl:
        pass
    print(batch_X.shape)  # type: ignore
   # print('batch_X:', batch_X)
    print('batch_y:', batch_y)
    print('batch_ind:', batch_ind)

    #pre_train_model(batch_size=100, num_workers=1,save_freq=10,Shuffel=False,save_dir_model='models',model_file_name='test',model_weights_dict=None,temperature= 2
    #               ,learning_rate= 0.01
    #               , weight_decay= 0.01,max_epochs=20,batch_print_condition=5)