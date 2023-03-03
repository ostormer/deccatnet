import pickle
from tqdm import tqdm
import time
from braindecode.datasets import BaseConcatDataset, BaseDataset, WindowsDataset
import braindecode.datasets.tuh as tuh
from braindecode.preprocessing import create_fixed_length_windows
from torch.utils.data import DataLoader
from mne import set_log_level
import mne
import braindecode
import torch
import torch.utils.data


def split_channels_and_window(concat_dataset:braindecode.datasets.BaseConcatDataset, channel_split_func=None, window_size_samples=2500) -> braindecode.datasets.BaseConcatDataset:
    """Splits BaseConcatDataset into set containing non-overlapping windows split into channels according to channel_split_func

    Args:
        concat_dataset (braindecode.datasets.BaseConcatDataset): Input dataset
        channel_split_func (callable, optional): Callable function f(ch_names:list[str]) -> list[list[str]]. If None, _make_overlapping_adjacent_pairs is used. Defaults to None.
        window_size_samples (int, optional): Number of time points per window. Defaults to 2500.

    Returns:
        braindecode.datasets.BaseConcatDataset: BaseConcatDataset containing WindowDatasets which have been split up into time windows and channel combinations
    """
    if channel_split_func is None:
        channel_split_func = _make_overlapping_adjacent_pairs
    for base_ds in concat_dataset.datasets:
        base_ds.raw.drop_channels(['IBI', 'BURSTS', 'SUPPR', 'PHOTIC PH'], on_missing='ignore')  # type: ignore
    # Windowing
    t0 = time.time()
    print(f"Begun windowing at {time.ctime(time.time())}")
    windowed_ds = create_fixed_length_windows(
        concat_dataset,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=window_size_samples,
        window_stride_samples=window_size_samples,
        drop_last_window=True,
    )
    # store the number of windows required for loading later on
    windowed_ds.set_description({
        "n_windows": [len(d) for d in windowed_ds.datasets]})  # type: ignore
    print(f'Finished windowing in {time.time()-t0} seconds')

    all_base_ds = []
    for windows_ds in tqdm(windowed_ds.datasets):
        split_concat_ds = _split_windows_into_channels(windows_ds, channel_split_func)  # type: ignore
        all_base_ds.append(split_concat_ds)
        
    final_ds = BaseConcatDataset(all_base_ds)
    # print(concat_ds.description)
    return final_ds


def _make_single_channels(ch_list:'list[str]') -> 'list[list[str]]':
    return [[ch] for ch in ch_list]

def _make_unique_pair_combos(ch_list:'list[str]') -> 'list[list[str]]':
    assert len(ch_list) > 1
    pairs = []
    for i, channel_i in enumerate(ch_list):
        for channel_j in ch_list[i+1:]:
            pairs.append([channel_i, channel_j])
    return pairs

def _make_all_pair_combos(ch_list:'list[str]') -> 'list[list[str]]':
    assert len(ch_list) > 1
    pairs = []
    for channel_i in ch_list:
        for channel_j in ch_list:
            pairs.append([channel_i, channel_j])
    return pairs

def _make_adjacent_pairs(ch_list:'list[str]') -> 'list[list[str]]':
    assert len(ch_list) > 1
    pairs = []
    for i in range(len(ch_list)//2):
        pairs.append([ch_list[2*i], ch_list[2*i + 1]])
    if len(ch_list) % 2 == 1:
        pairs.append([ch_list[-1], ch_list[-2]])
    return pairs

def _make_overlapping_adjacent_pairs(ch_list:'list[str]') -> 'list[list[str]]':
    assert len(ch_list) > 1
    pairs = []
    for i in range(len(ch_list) - 1):
        pairs.append([ch_list[i], ch_list[i+1]])    
    return pairs

def _split_windows_into_channels(base_ds:braindecode.datasets.WindowsDataset, channel_split_func=_make_single_channels) -> braindecode.datasets.BaseConcatDataset:
    raw = base_ds.windows._raw
    channel_selections = channel_split_func(raw.ch_names)

    base_ds_list = []
    old_epochs = base_ds.windows
    raw.load_data()
    for channels in channel_selections:
        new_epochs = mne.Epochs(
            raw=raw,
            events=old_epochs.events,
            picks=channels,
            baseline=None,
            tmin=0,
            tmax=old_epochs.tmax,
            metadata=old_epochs.metadata
        )
        new_epochs.drop_bad()

        ds = WindowsDataset(new_epochs, base_ds.description)
        ds.set_description({"channels": channels})
        base_ds_list.append(ds)
        
    concat = BaseConcatDataset(base_ds_list)
    return concat


if __name__ == "__main__":
    READ_CACHED_DS = False  # Change to read cache or not
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
    single_channel_windows = split_channels_and_window(dataset, channel_split_func=_make_overlapping_adjacent_pairs)

    with open('windowed_test.pkl', 'wb') as f:
        pickle.dump(single_channel_windows, f)

    print("Windowing complete")
    # Default DataLoader object lets us iterate through the dataset.
    # Each call to get item returns a batch of samples,
    # each batch has shape: (batch_size, n_channels, n_time_points)
    loader = DataLoader(dataset=single_channel_windows, batch_size=32)

    batch_X, batch_y, batch_ind = None, None, None
    i = 0
    for batch_X, batch_y, batch_ind in tqdm(loader):
        if i < 1:
            i+=1
            print(batch_X.shape)  # type: ignore
            print('batch_X:', batch_X)
            print('batch_y:', batch_y)
            print('batch_ind:', batch_ind)
        pass
    print(batch_X.shape)  # type: ignore
    print('batch_X:', batch_X)
    print('batch_y:', batch_y)
    print('batch_ind:', batch_ind)
