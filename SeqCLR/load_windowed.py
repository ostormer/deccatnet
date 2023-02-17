from braindecode.datasets import BaseConcatDataset, BaseDataset
from mne import Epochs
from mne.io import read_raw_edf
import pickle
from tqdm import tqdm
import os
import braindecode.datasets.tuh as tuh


# def walk_sub_dirs(path, function) -> list:
#     """Walk subdir tree of path, apply 'function' to all .edf files.
#     Return values are appended to a list which is returned

#     Args:
#         path (string): Top level of subdir tree to walk
#         function (function): Function to apply, taking the absolute file path as the only argument

#     Returns:
#         list: List of values returned by function on all edf files
#     """
#     n_files = 0
#     for dir_path, _dir_names, file_names in os.walk(path):
#         for f in file_names:
#             if f.endswith(".edf"):
#                 n_files += 1
#     print(f"Applying function {function.__name__} to {n_files} edf files...")
#     returned = []
#     with tqdm(total=n_files) as progress_bar:
#         for dir_path, _dir_names, file_names in tqdm(os.walk(path)):
#             for f in file_names:
#                 if f.endswith(".edf"):
#                     progress_bar.update(1)
#                     file_path = os.path.join(dir_path, f)
#                     returned.append(function(file_path))
#     print("Done!")
#     return returned


# def read_raw_edf_wrapper(file_path):
#     return read_raw_edf(file_path, verbose='WARNING')



if __name__ == "__main__":
    READ_CACHED_DS = True
    # DATASET_ROOT = 'D:/TUH/tuh_eeg_abnormal'
    # CACHE_PATH = 'datasets/tuh_braindecode/tuh_abnormal.pkl'
    DATASET_ROOT = 'datasets/tuh-test/tuh-eeg'
    CACHE_PATH = 'datasets/tuh_braindecode/tuh_eeg.pkl'

    if READ_CACHED_DS:
        with open(CACHE_PATH, 'rb') as f:
            # ds_abnormal = pickle.load(f)
            ds = pickle.load(f)
    else:
        # ds_abnormal = tuh.TUHAbnormal(DATASET_ROOT)
        ds = tuh.TUH(DATASET_ROOT)

        with open(CACHE_PATH, 'wb') as f:
            # pickle.dump(ds_abnormal, f)
            pickle.dump(ds, f)
    
    print(ds.description)