from braindecode.datasets import BaseConcatDataset, BaseDataset
from mne import Epochs
from mne.io import read_raw_edf
import pickle as pkl
from tqdm import tqdm
import os
import braindecode.datasets.tuh as tuh


def walk_sub_dirs(path, function) -> list:
    """Walk subdir tree of path, apply 'function' to all .edf files.
    Return values are appended to a list which is returned

    Args:
        path (string): Top level of subdir tree to walk
        function (function): Function to apply, taking the absolute file path as the only argument

    Returns:
        list: List of values returned by function on all edf files
    """
    n_files = 0
    for dir_path, _dir_names, file_names in os.walk(path):
        for f in file_names:
            if f.endswith(".edf"):
                n_files += 1
    print(f"Applying function {function.__name__} to {n_files} edf files...")
    returned = []
    with tqdm(total=n_files) as progress_bar:
        for dir_path, _dir_names, file_names in tqdm(os.walk(path)):
            for f in file_names:
                if f.endswith(".edf"):
                    progress_bar.update(1)
                    file_path = os.path.join(dir_path, f)
                    returned.append(function(file_path))
    print("Done!")
    return returned


def read_raw_edf_wrapper(file_path):
    return read_raw_edf(file_path, verbose='WARNING')



if __name__ == "__main__":
    
    DATASET_ROOT = "D:/TUH/tuh_eeg/v2.0.0/edf/000"
    records = walk_sub_dirs(DATASET_ROOT, read_raw_edf_wrapper)
    print("\n", records[:5])

    # epochs = [Epochs(raw) ]
