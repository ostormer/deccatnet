import pickle
from tqdm import tqdm
import os
import braindecode.datasets.tuh as tuh
from braindecode.preprocessing import create_fixed_length_windows
from torch.utils.data import DataLoader

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
    read_cached_ds = True
    load_abnormal = True

    DATASET_ROOT = None
    CACHE_PATH = None
    dataset = None
    if load_abnormal:
        DATASET_ROOT = 'D:/TUH/tuh_test/tuh_eeg_abnormal'
        CACHE_PATH = 'datasets/tuh_braindecode/tuh_abnormal.pkl'

    else:
        DATASET_ROOT = 'datasets/tuh_test/tuh_eeg'
        CACHE_PATH = 'datasets/tuh_braindecode/tuh_eeg.pkl'

    if read_cached_ds:
        with open(CACHE_PATH, 'rb') as f:
            
            dataset = pickle.load(f)
    else:
        if load_abnormal:
            ds_abnormal = tuh.TUHAbnormal(DATASET_ROOT)
        else:
            dataset = tuh.TUH(DATASET_ROOT)

        with open(CACHE_PATH, 'wb') as f:
            # pickle.dump(ds_abnormal, f)
            pickle.dump(dataset, f)
    
    # print(ds.description)

    subset = dataset.split(by=range(10))['0']
    print(subset.description)

    subset_windows = create_fixed_length_windows(
        subset,
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
    
    dl = DataLoader(dataset=subset_windows, batch_size=4)

    

    batch_X, batch_y, batch_ind = None, None, None
    for batch_X, batch_y, batch_ind in dl:
        pass
    print('batch_X:', batch_X)
    print('batch_y:', batch_y)
    print('batch_ind:', batch_ind)