import bisect
import pickle

import numpy as np
from tqdm import tqdm
import time
from braindecode.datasets import BaseConcatDataset, WindowsDataset
import braindecode.datasets.tuh as tuh
from braindecode.preprocessing import create_fixed_length_windows
from torch.utils.data import DataLoader
from mne import set_log_level
import mne


# def split_channels_and_window_2(concat_dataset:BaseConcatDataset, channel_split_func=None, window_size_samples=2500) -> BaseConcatDataset:
#     """Splits BaseConcatDataset into set containing non-overlapping windows split into channels according to channel_split_func

#     Args:
#         concat_dataset (braindecode.datasets.BaseConcatDataset): Input dataset
#         channel_split_func (callable, optional): Callable function f(ch_names:list[str]) -> list[list[str]]. If None, _make_overlapping_adjacent_pairs is used. Defaults to None.
#         window_size_samples (int, optional): Number of time points per window. Defaults to 2500.

#     Returns:
#         braindecode.datasets.BaseConcatDataset: BaseConcatDataset containing WindowDatasets which have been split up into time windows and channel combinations
#     """
#     if channel_split_func is None:
#         channel_split_func = _make_overlapping_adjacent_pairs
#     # Windowing
#     t0 = time.time()
#     print(f"Begun windowing at {time.ctime(time.time())}")
#     windowed_sets = []
#     for base_ds in tqdm(concat_dataset.datasets):
#         base_ds.raw.drop_channels(['IBI', 'BURSTS', 'SUPPR', 'PHOTIC PH'], on_missing='ignore')  # type: ignore
#         picks = channel_split_func(base_ds.raw.ch_names)  # type: ignore
#         print(picks)
#         for pick in picks:
#             single_windowed_ds = create_fixed_length_windows(
#                 concat_dataset,
#                 picks=pick,
#                 start_offset_samples=0,
#                 stop_offset_samples=None,
#                 window_size_samples=window_size_samples,
#                 window_stride_samples=window_size_samples,
#                 drop_last_window=True,
#             )
#             if len(single_windowed_ds.datasets[0].windows.ch_names) != len(pick):
#                 continue
#             # store the number of windows required for loading later on
#             single_windowed_ds.set_description({
#                 "n_windows": [len(d) for d in single_windowed_ds.datasets]})  # type: ignore
#             windowed_sets.append(single_windowed_ds)
#     print(f'Finished windowing in {time.time()-t0} seconds')

#     final_ds = BaseConcatDataset(windowed_sets)
#     # print(concat_ds.description)
#     return final_ds

"""
Plan and things which needs to be done:
For one and one file:
- preprocess, split into windows and parallelize(split into channels) and then finally add some more preprocessing. 
    This may take several days, but it doesnt matter
- window and split into channels
- save data in a file structure

After this:
- read new data to a dataset
- create dataset which is where we apply augmentations

Two main approaches: use first 11 and remove first and then use 21 minutes, drop last window. 
windows size = 60s


"""
def select_duration(concat_ds:BaseConcatDataset, t_min=0, t_max=None):
    """
    Selects all recordings which fulfills: t_min < duration < t_max, data is not loaded here, only metadata
    :param ds: dataset
    :param t_min: mimimum length of recordings
    :param t_max: maximum length of recordings
    :return: satisfactory recordings
    """
    if t_max == None:
        t_max = np.Inf
    idx = []
    test = []
    for i,ds in tqdm(enumerate(concat_ds.datasets)):
        len_session = ds.raw.tmax - ds.raw.tmin
        duration = ds.raw.n_times / ds.raw.info['sfreq']
        assert round(len_session) == round(duration)
        test.append(len_session)
        if len_session > t_min and len_session < t_max:
            idx.append(i)
    good_recordings = [concat_ds.datasets[i] for i in idx]
    return BaseConcatDataset(good_recordings)

def rename_channels(raw,mapping):
    """
    Renames all channels such that the naming conventions is similar for all recordings
    :param concat_ds: dataset
    :param mapping: a mapping object which maps a reference to
    :return: ds where similar recordings has similar names
    """
    concat_ds.datasets[0].raw.rename_channels

    # now we will select which ch_names we are interested in

def get_unique_channel_names(concat_ds:BaseConcatDataset):
    unique_ch_names = []
    for i,ds in tqdm(enumerate(concat_ds.datasets)):
        unique_ch_names.extend(list(set(ds.raw.info.ch_names) - set(unique_ch_names)))
    print(unique_ch_names)
    le_channels = ['EEG 28-LE', 'EEG T1-LE', 'EEG T3-LE', 'EEG O1-LE', 'EEG T6-LE', 'EEG F3-LE', 'EEG F4-LE', 'EEG F8-LE',
     'EEG A1-LE', 'EEG PG1-LE', 'EEG SP2-LE', 'EEG T2-LE', 'EEG T5-LE', 'EEG A2-LE', 'EEG OZ-LE', 'EEG F7-LE',
     'EEG FP1-LE', 'EEG P3-LE', 'EEG PG2-LE', 'EEG SP1-LE', 'EEG FZ-LE', 'EEG C3-LE', 'EEG T4-LE', 'EEG PZ-LE',
     'EEG 29-LE', 'EEG EKG-LE', 'PHOTIC PH', 'EEG P4-LE', 'EEG CZ-LE', 'EEG O2-LE', 'EEG C4-LE', 'EEG FP2-LE',
     'EEG 30-LE', 'EEG RLC-LE', 'EEG LUC-LE''EEG 31-LE', 'EEG 27-LE', 'EEG 26-LE', 'EEG 32-LE','EEG 21-LE', 'EEG 24-LE', 'EEG 22-LE', 'EEG 25-LE', 'EEG 20-LE', 'EEG 23-LE']

    ar_channels = ['EEG A1-REF', 'EEG T3-REF', 'EEG F7-REF', 'EEG F4-REF', 'EEG O1-REF',
     'EEG FP1-REF', 'EEG PZ-REF', 'EEG T4-REF', 'EEG CZ-REF', 'EEG FP2-REF', 'EEG RESP2-REF', 'EEG A2-REF',
     'EEG T2-REF', 'EEG SP1-REF', 'EEG RESP1-REF', 'EEG C3-REF', 'EEG F8-REF', 'EEG RLC-REF', 'EEG FZ-REF',
     'EEG P4-REF', 'EEG EKG-REF', 'EEG C4-REF', 'EEG T1-REF', 'EEG SP2-REF', 'EEG 31-REF', 'EEG O2-REF', 'EEG T6-REF',
     'EEG F3-REF', 'EEG LUC-REF', 'EEG P3-REF', 'EEG 32-REF', 'EEG T5-REF', 'EEG 81-REF', 'EEG 99-REF', 'EEG 27-REF',
     'EEG 103-REF', 'EEG 22-REF', 'EEG 33-REF', 'EEG 43-REF', 'EEG 23-REF', 'EEG 78-REF', 'EEG 59-REF', 'EEG 121-REF',
     'EEG 28-REF', 'EEG 104-REF', 'EEG 93-REF', 'EEG 84-REF', 'EEG 111-REF', 'EEG 71-REF', 'EEG 52-REF', 'EEG 115-REF',
     'EEG 117-REF', 'EEG 114-REF', 'EEG 34-REF', 'EEG 123-REF', 'EEG 58-REF', 'EEG 76-REF', 'EEG 47-REF', 'EEG 69-REF',
     'EEG 74-REF', 'EEG 95-REF', 'EEG 112-REF', 'EEG 70-REF', 'EEG 37-REF', 'EEG 64-REF', 'EEG 87-REF', 'EEG 122-REF',
     'EEG 120-REF', 'EEG 124-REF', 'EEG 29-REF', 'EEG 44-REF', 'EEG 125-REF', 'EEG 73-REF', 'EEG 46-REF', 'EEG 82-REF',
     'EEG 35-REF', 'EEG 30-REF', 'EEG 66-REF', 'EEG 118-REF', 'EEG 67-REF', 'EEG 79-REF', 'EEG 119-REF', 'EEG 108-REF',
     'EEG 88-REF', 'EEG 63-REF', 'EEG 61-REF', 'EEG 127-REF', 'EEG 98-REF', 'EEG 54-REF', 'EEG 110-REF', 'EEG 25-REF',
     'EEG 50-REF', 'EEG 56-REF', 'EEG 24-REF', 'EEG 97-REF', 'EEG 107-REF', 'EEG 65-REF', 'EEG 106-REF', 'EEG 51-REF',
     'EEG 102-REF', 'EEG 109-REF', 'EEG 100-REF', 'EEG 21-REF', 'EEG 91-REF', 'EEG 89-REF', 'EEG 86-REF', 'EEG 20-REF',
     'EEG 57-REF', 'EEG 49-REF', 'EEG 83-REF', 'EEG 96-REF', 'EEG 77-REF', 'EEG 40-REF', 'EEG 72-REF', 'EEG 94-REF',
     'EEG 80-REF', 'EEG 41-REF', 'EEG 128-REF', 'EEG 75-REF', 'EEG 39-REF', 'EEG 92-REF', 'EEG 101-REF', 'EEG 48-REF',
     'EEG 42-REF', 'EEG 45-REF', 'EEG 68-REF', 'EEG 126-REF', 'EEG 90-REF', 'EEG 85-REF', 'EEG 53-REF', 'EEG 62-REF',
     'EEG 55-REF', 'EEG 38-REF', 'EEG 26-REF', 'EEG 36-REF', 'EEG 60-REF', 'EEG 113-REF', 'EEG 116-REF', 'EEG 105-REF','EEG ROC-REF'
                   ,'EEG EKG1-REF','EEG C3P-REF', 'EEG C4P-REF',
     'EEG OZ-REF','EEG LOC-REF']

    not_interesting = ['DC4-DC', 'DC3-DC', 'DC7-DC', 'DC2-DC', 'DC8-DC', 'DC6-DC',
     'DC1-DC', 'DC5-DC',
      'EMG-REF', 'SUPPR', 'IBI', 'PHOTIC-REF', 'BURSTS' , 'ECG EKG-REF', 'PULSE RATE', 'RESP ABDOMEN-REF']


def first_preprocess_step(concat_dataset:BaseConcatDataset, n_jobs):
    """
    renames channels to common nameing, resamples all data to one frequency, sets common eeg_reference, applies
    bandpass filter and crops.
    :param concat_dataset: a TUH BaseConcatDataset which
    :param n_jobs: number of aviables jobs for parallelization
    :return: preprocessed BaseConcatDataset
    """



def split_channels_and_window(concat_dataset:BaseConcatDataset, channel_split_func=None, window_size_samples=2500) -> BaseConcatDataset:
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
        n_jobs=4,
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


def _split_windows_into_channels(base_ds:WindowsDataset, channel_split_func=_make_single_channels) -> BaseConcatDataset:
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

def _split_windows_into_channels(base_ds:WindowsDataset, channel_split_func=_make_single_channels) -> BaseConcatDataset:
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
    SOURCE_DS = 'tuh_eeg_abnormal'  # Which dataset to load

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
