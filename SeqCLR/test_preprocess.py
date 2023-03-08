import bisect
import pickle
from tqdm import tqdm
import time
from braindecode.datasets import BaseConcatDataset, WindowsDataset
import braindecode.datasets.tuh as tuh
from braindecode.preprocessing import create_fixed_length_windows
from torch.utils.data import DataLoader
from mne import set_log_level
import mne
from load_windowed import select_duration, rename_channels, get_unique_channel_names

if __name__ == "__main__":
    READ_CACHED_DS = True  # Change to read cache or not
    SOURCE_DS = 'tuh_eeg'  # Which dataset to load

    assert SOURCE_DS in ['tuh_eeg_abnormal', 'tuh_eeg']
    # Disable most MNE logging output which slows execution
    set_log_level(verbose='WARNING')

    dataset_root = None
    cache_path = None
    dataset = None
    if SOURCE_DS == 'tuh_eeg_abnormal':
        dataset_root = 'datasets/tuh_test/tuh_eeg_abnormal'
        cache_path = 'datasets/tuh_braindecode/tuh_abnormal.pkl'

    else:
        dataset_root = 'D:/TUH/tuh_eeg'
        cache_path = 'D:/TUH/pickles/tuh_eeg'

    if READ_CACHED_DS:
        with open(cache_path, 'rb') as f:

            dataset = pickle.load(f)
            print('done loading pickled dataset')
    else:

        if SOURCE_DS == 'tuh_eeg_abnormal':
            dataset = tuh.TUHAbnormal(dataset_root)

        else:
            dataset = tuh.TUH(dataset_root, n_jobs=2)
            print('done creating TUH dataset')
        with open(cache_path, 'wb') as f:
            pickle.dump(dataset, f)
        print('done pickling')



    dataset = select_duration(dataset,t_min=10, t_max=1000)
    dataset = get_unique_channel_names(dataset)
