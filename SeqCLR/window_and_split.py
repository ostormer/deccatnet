import mne
import os, shutil, itertools
import pickle
import braindecode.datasets.tuh as tuh
from tqdm import tqdm
from copy import deepcopy
from braindecode.datasets import BaseConcatDataset, BaseDataset, WindowsDataset
from braindecode.preprocessing import create_fixed_length_windows
from joblib import Parallel, delayed
from braindecode.datautil.serialization import load_concat_dataset, _check_save_dir_empty


def window_and_split(concat_ds: BaseConcatDataset, save_dir: str, overwrite=False,
                     window_size_samples=5000, n_jobs=1, channel_split_func=None) -> 'list[int]':
    if channel_split_func is None:
        channel_split_func = make_adjacent_pairs
    # Drop too short samples
    concat_ds.set_description({"n_samples": [ds.raw.n_times for ds in concat_ds.datasets]})  # type: ignore
    keep = [n >= window_size_samples for n in concat_ds.description["n_samples"]]
    keep_indexes = [i for i, k in enumerate(keep) if k == True]
    concat_ds = concat_ds.split(by=keep_indexes)["0"]
    print("WINDOWING DATASET")
    windows_ds = create_fixed_length_windows(
        concat_ds,
        n_jobs=n_jobs,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=window_size_samples,
        window_stride_samples=window_size_samples,
        drop_last_window=True,
        drop_bad_windows=True,
        verbose='ERROR'
    )
    # Prepare save dir
    save_dir = os.path.abspath(save_dir)
    if not overwrite:
        _check_save_dir_empty(save_dir)
    # Create save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Delete old contents of save_dir
    for files in os.listdir(save_dir):
        path = os.path.join(save_dir, files)
        try:
            shutil.rmtree(path)
        except OSError:
            os.remove(path)
    # Save pickle of windows_dataset
    os.makedirs(os.path.join(save_dir, "pickles"))
    with open(os.path.join(save_dir, "pickles/windowed.pkl"), 'wb') as file:
        pickle.dump(windows_ds, file)

    # Prepare for channel splitting
    os.makedirs(os.path.join(save_dir, "fif_ds"))
    channel_split_dir = os.path.join(save_dir, "fif_ds")
    print('Splitting recordings into separate channels')
    indexes = Parallel(n_jobs=n_jobs)(
        delayed(_split_channels)(windows_ds, i, channel_split_dir, channel_split_func)
        for i, windows_ds in tqdm(enumerate(windows_ds.datasets), total=len(windows_ds.datasets))
    )
    print('Reloading serialized dataset')
    indexes = itertools.chain.from_iterable(indexes)  # type: ignore

    return indexes  # type: ignore


def _split_channels(windows_ds: WindowsDataset, record_index: int, save_dir: str, channel_split_func) -> 'list[int]':
    mne.set_log_level(verbose='ERROR')
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
        ds.window_kwargs = deepcopy(windows_ds.window_kwargs)  # type: ignore
        print(ds.window_kwargs)
        ds.set_description({"channels": channels})
        windows_ds_list.append(ds)
    # Serialization:
    # Create new BaseConcatDataset from each dataset, and save it to disk.
    # Then it can be unloaded from memory
    # OLD: ----------------------------
    '''
    rec_id = "{:}_s{:02d}_t{:02d}".format(
        windows_ds.description["subject"],
        windows_ds.description["session"],
        windows_ds.description["segment"])
    save_path = os.path.join(save_dir, rec_id)
    if not os.path.exists(save_path):
    os.makedirs(save_path)

    concat_ds = BaseConcatDataset(windows_ds_list)
    concat_ds.save(save_path, overwrite=True)

    return save_path
    '''
    # NEW:
    concat_ds = BaseConcatDataset(windows_ds_list)
    concat_ds.save(save_dir, overwrite=True, offset=record_index*100)
    indexes = list(range(record_index*100, record_index*100+len(windows_ds_list)))
    return indexes


def make_single_channels(ch_list: 'list[str]') -> 'list[list[str]]':
    return [[ch] for ch in ch_list]

def make_unique_pair_combos(ch_list: 'list[str]') -> 'list[list[str]]':
    assert len(ch_list) > 1
    pairs = []
    for i, channel_i in enumerate(ch_list):
        for channel_j in ch_list[i+1:]:
            pairs.append([channel_i, channel_j])
    return pairs

def make_all_pair_combos(ch_list: 'list[str]') -> 'list[list[str]]':
    assert len(ch_list) > 1
    pairs = []
    for channel_i in ch_list:
        for channel_j in ch_list:
            pairs.append([channel_i, channel_j])
    return pairs

def make_adjacent_pairs(ch_list: 'list[str]') -> 'list[list[str]]':
    assert len(ch_list) > 1
    pairs = []
    for i in range(len(ch_list)//2):
        pairs.append([ch_list[2*i], ch_list[2*i + 1]])
    if len(ch_list) % 2 == 1:
        pairs.append([ch_list[-1], ch_list[-2]])
    return pairs

def make_overlapping_adjacent_pairs(ch_list: 'list[str]') -> 'list[list[str]]':
    assert len(ch_list) > 1
    pairs = []
    for i in range(len(ch_list) - 1):
        pairs.append([ch_list[i], ch_list[i+1]])
    return pairs

if __name__ == "__main__":
    READ_CACHED_DS = True  # Change to read cache or not
    SOURCE_DS = 'tuh_eeg'  # Which dataset to load
    SOURCE_DS_ROOT = 'D:/TUH/tuh_eeg'


    assert SOURCE_DS in ['tuh_eeg_abnormal', 'tuh_eeg']
    # Disable most MNE logging output which slows execution
    mne.set_log_level(verbose='ERROR')

    dataset_root = SOURCE_DS_ROOT
    cache_path = 'D:/TUH/pickles/tuh_eeg'
    dataset = None
    # if SOURCE_DS == 'tuh_eeg_abnormal':
    #     dataset_root = 'datasets/tuh_test/tuh_eeg_abnormal'
    #     cache_path = 'datasets/tuh_braindecode/tuh_abnormal.pkl'

    # else:
    #     dataset_root = 'datasets/tuh_test/tuh_eeg'
    #     cache_path = 'datasets/tuh_braindecode/tuh_eeg.pkl'

    if READ_CACHED_DS:
        print(f"Reading cached ds from path: {cache_path}")
        with open(cache_path, 'rb') as f:
            dataset = pickle.load(f)
    else:
        print(f"Reading {SOURCE_DS} from path: {dataset_root}")
        if SOURCE_DS == 'tuh_eeg_abnormal':
            dataset = tuh.TUHAbnormal(dataset_root)
        else:
            dataset = tuh.TUH(dataset_root)

        with open(cache_path, 'wb') as f:
            pickle.dump(dataset, f)

    print('Loaded DS')

    # dataset = dataset.split(by=range(10))['0']

    ids_to_load = window_and_split(
        dataset, "d:/TUH/preprocessed", overwrite=True, n_jobs=8)
    
    with open('datasets/tuh_test/tuh_split_indexes.pkl', 'wb') as fd:
        pickle.dump(ids_to_load, fd)

