import mne
import os
import itertools
import pickle
import warnings
import braindecode.datasets.tuh as tuh
from braindecode.datasets import BaseConcatDataset, BaseDataset, WindowsDataset
from braindecode.preprocessing import create_fixed_length_windows
from joblib import Parallel, delayed
from braindecode.datautil.serialization import load_concat_dataset, _check_save_dir_empty


def split_and_window(concat_ds: BaseConcatDataset, save_dir: str, overwrite=False,
                     window_size_samples=5000, n_jobs=1, channel_split_func=None) -> BaseConcatDataset:
    if channel_split_func is None:
        channel_split_func = _make_adjacent_pairs
    # Drop too short samples
    concat_ds.set_description({"n_samples": [ds.raw.n_times for ds in concat_ds.datasets]})
    keep = [n >= window_size_samples for n in concat_ds.description["n_samples"]]
    keep_indexes = [i for i, k in enumerate(keep) if k == True]
    concat_ds = concat_ds.split(by=keep_indexes)["0"]
    print("START WINDOWING")
    windows_ds = create_fixed_length_windows(
        concat_ds,
        n_jobs=n_jobs,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=window_size_samples,
        window_stride_samples=window_size_samples,
        drop_last_window=True,
        drop_bad_windows=True,

    )
    print("DONE WONDOWING")
    # Prepare save dir
    save_dir = os.path.abspath(save_dir)
    if not overwrite:
        _check_save_dir_empty(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # TODO: Delete old?

    subset_paths = Parallel(n_jobs=n_jobs)(
        delayed(_split_channels)(windows_ds, i, save_dir, channel_split_func)
        for i, windows_ds in enumerate(windows_ds.datasets)
    )
    subsets = Parallel(n_jobs=n_jobs)(
        delayed(load_concat_dataset)(subset_path, preload=False, n_jobs=1)
        for subset_path in subset_paths
    )
    concat_ds = BaseConcatDataset(subsets)
    # concat_ds = load_concat_dataset(save_dir, preload=False, n_jobs=n_jobs, ids_to_load=indexes)
    print(concat_ds.description)
    return concat_ds


def _split_channels(windows_ds: WindowsDataset, record_index: int, save_dir: str, channel_split_func) -> str:
    mne.set_log_level("ERROR")
    raw = windows_ds.windows._raw
    raw.drop_channels(['IBI', 'BURSTS', 'SUPPR', 'PHOTIC PH'],
                      on_missing='ignore')  # type: ignore
    channel_selections = channel_split_func(raw.ch_names)
    windows_ds_list = []
    raw.load_data()
    for channels in channel_selections:
        new_epochs = mne.Epochs(
            raw=raw,
            events=windows_ds.windows.events,
            picks=channels,
            baseline=None,
            tmin=0,
            tmax=windows_ds.windows.tmax,
            metadata=windows_ds.windows.metadata
        )
        new_epochs.drop_bad()

        ds = WindowsDataset(new_epochs, windows_ds.description)
        ds.set_description({"channels": channels})
        windows_ds_list.append(ds)
    # Serialization:
    # Create new BaseConcatDataset from each dataset, and save it to disk.
    # Then it can be unloaded from memory
    rec_id = "{:}_s{:02d}_t{:02d}".format(
        windows_ds.description["subject"],
        windows_ds.description["session"],
        windows_ds.description["segment"])
    save_path = os.path.join(save_dir, rec_id)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # indexes = list(range(record_index*100, record_index*100+len(windows_ds_list)))
    concat_ds = BaseConcatDataset(windows_ds_list)
    concat_ds.save(save_path, overwrite=True)

    return save_path


def _make_single_channels(ch_list: 'list[str]') -> 'list[list[str]]':
    return [[ch] for ch in ch_list]

def _make_unique_pair_combos(ch_list: 'list[str]') -> 'list[list[str]]':
    assert len(ch_list) > 1
    pairs = []
    for i, channel_i in enumerate(ch_list):
        for channel_j in ch_list[i+1:]:
            pairs.append([channel_i, channel_j])
    return pairs

def _make_all_pair_combos(ch_list: 'list[str]') -> 'list[list[str]]':
    assert len(ch_list) > 1
    pairs = []
    for channel_i in ch_list:
        for channel_j in ch_list:
            pairs.append([channel_i, channel_j])
    return pairs

def _make_adjacent_pairs(ch_list: 'list[str]') -> 'list[list[str]]':
    assert len(ch_list) > 1
    pairs = []
    for i in range(len(ch_list)//2):
        pairs.append([ch_list[2*i], ch_list[2*i + 1]])
    if len(ch_list) % 2 == 1:
        pairs.append([ch_list[-1], ch_list[-2]])
    return pairs

def _make_overlapping_adjacent_pairs(ch_list: 'list[str]') -> 'list[list[str]]':
    assert len(ch_list) > 1
    pairs = []
    for i in range(len(ch_list) - 1):
        pairs.append([ch_list[i], ch_list[i+1]])
    return pairs

if __name__ == "__main__":
    READ_CACHED_DS = False  # Change to read cache or not
    SOURCE_DS = 'tuh_eeg'  # Which dataset to load

    assert SOURCE_DS in ['tuh_eeg_abnormal', 'tuh_eeg']
    # Disable most MNE logging output which slows execution
    mne.set_log_level(verbose='ERROR')

    dataset_root = None
    cache_path = None
    dataset = None
    if SOURCE_DS == 'tuh_eeg_abnormal':
        dataset_root = 'datasets/tuh_test/tuh_eeg_abnormal'
        cache_path = 'datasets/tuh_braindecode/tuh_abnormal.pkl'

    else:
        dataset_root = 'datasets/tuh_test/tuh_eeg'
        cache_path = 'datasets/tuh_braindecode/tuh_eeg.pkl'

    if READ_CACHED_DS:
        with open(cache_path, 'rb') as f:

            dataset = pickle.load(f)
    else:

        if SOURCE_DS == 'tuh_eeg_abnormal':
            dataset = tuh.TUHAbnormal(dataset_root)

        else:
            dataset = tuh.TUH(dataset_root)

        with open(cache_path, 'wb') as f:
            pickle.dump(dataset, f)

    print(dataset.description)
    print('Loaded DS')

    dataset = dataset.split(by=range(10))['0']

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        windowed = split_and_window(
            dataset, "datasets/tuh_braindecode/tuh_split", overwrite=True, n_jobs=4)
    

    print(windowed.description)
