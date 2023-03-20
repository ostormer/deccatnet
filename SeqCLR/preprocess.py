import pickle
import os
import numpy as np
import shutil
import mne
from copy import deepcopy
import itertools

from copy import deepcopy
from tqdm import tqdm
from braindecode.datasets import BaseConcatDataset, WindowsDataset
from braindecode.preprocessing import create_fixed_length_windows, Preprocessor, preprocess, scale
from braindecode.datautil.serialization import load_concat_dataset, _check_save_dir_empty
from joblib import Parallel, delayed


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
le_channels = ['EEG 28-LE', 'EEG T1-LE', 'EEG T3-LE', 'EEG O1-LE', 'EEG T6-LE', 'EEG F3-LE', 'EEG F4-LE', 'EEG F8-LE',
               'EEG A1-LE', 'EEG PG1-LE', 'EEG SP2-LE', 'EEG T2-LE', 'EEG T5-LE', 'EEG A2-LE', 'EEG OZ-LE', 'EEG F7-LE',
               'EEG FP1-LE', 'EEG P3-LE', 'EEG PG2-LE', 'EEG SP1-LE', 'EEG FZ-LE', 'EEG C3-LE', 'EEG T4-LE', 'EEG PZ-LE',
               'EEG 29-LE', 'EEG EKG-LE', 'PHOTIC PH', 'EEG P4-LE', 'EEG CZ-LE', 'EEG O2-LE', 'EEG C4-LE', 'EEG FP2-LE',
               'EEG 30-LE', 'EEG RLC-LE', 'EEG LUC-LE', 'EEG 31-LE', 'EEG 27-LE', 'EEG 26-LE', 'EEG 32-LE', 'EEG 21-LE',
               'EEG 24-LE', 'EEG 22-LE', 'EEG 25-LE', 'EEG 20-LE', 'EEG 23-LE']

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
               'EEG 55-REF', 'EEG 38-REF', 'EEG 26-REF', 'EEG 36-REF', 'EEG 60-REF', 'EEG 113-REF', 'EEG 116-REF', 'EEG 105-REF',
               'EEG ROC-REF', 'EEG EKG1-REF', 'EEG C3P-REF', 'EEG C4P-REF',
               'EEG OZ-REF', 'EEG LOC-REF']

not_interesting = ['DC4-DC', 'DC3-DC', 'DC7-DC', 'DC2-DC', 'DC8-DC', 'DC6-DC', 'DC1-DC', 'DC5-DC',
                   'EMG-REF', 'SUPPR', 'IBI', 'PHOTIC-REF', 'BURSTS', 'ECG EKG-REF', 'PULSE RATE', 'RESP ABDOMEN-REF']


def select_duration(concat_ds: BaseConcatDataset, t_min=0, t_max=None):
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
    for i, ds in enumerate(concat_ds.datasets):
        len_session = ds.raw.tmax - ds.raw.tmin  # type: ignore
        duration = ds.raw.n_times / ds.raw.info['sfreq']  # type: ignore
        assert round(len_session) == round(duration)
        test.append(len_session)
        if len_session > t_min and len_session < t_max:
            idx.append(i)
    good_recordings = [concat_ds.datasets[i] for i in idx]
    return BaseConcatDataset(good_recordings)


def rename_channels(raw, mapping):
    """
    Renames all channels such that the naming conventions is similar for all recordings
    This will  work as a custom preprocessor functions form  braindecode preprocessors
    :param concat_ds: dataset
    :param mapping: a mapping object which maps a reference to
    :return: ds where similar recordings has similar names
    """
    ref = raw.ch_names[0].split('-')[-1].lower()
    assert ref in ['le', 'ref'], "not valid reference for this experiment"
    ref = 'ar' if ref != 'le' else 'le'  # change from ref to ar if not le
    # get valid channels
    mapping = rewrite_mapping(raw.ch_names, mapping[ref])
    raw.rename_channels(mapping)


def rewrite_mapping(ch_names, mapping):
    """

    :param ch_names:
    :param mapping:
    :return:
    """

    ch_names = list(set(ch_names) & set(mapping.keys()))
    new_dict = {key: mapping[key] for key in ch_names}
    return new_dict


def get_unique_channel_names(concat_ds: BaseConcatDataset):
    unique_ch_names = []
    for i, ds in enumerate(concat_ds.datasets):
        unique_ch_names.extend(
            list(set(ds.raw.info.ch_names) - set(unique_ch_names)))  # type: ignore
    print(unique_ch_names)


def custom_turn_off_log(raw, verbose='ERROR'):
    mne.set_log_level(verbose)
    return raw


def first_preprocess_step(concat_dataset: BaseConcatDataset, mapping, ch_name, crop_min, crop_max, sfreq, save_dir, n_jobs):
    """
    renames channels to common nameing, resamples all data to one frequency, sets common eeg_reference, applies
    bandpass filter and crops.
    :param concat_dataset: a TUH BaseConcatDataset which
    :param n_jobs: number of aviables jobs for parallelization
    :return: preprocessed BaseConcatDataset
    """
    mne.set_log_level('ERROR')
    preprocessors = [Preprocessor(custom_turn_off_log),  # turn off verbose
                     # set common reference for all
                     Preprocessor('set_eeg_reference',
                                  ref_channels='average', ch_type='eeg'),
                     # rename to common naming convention
                     Preprocessor(rename_channels, mapping=mapping,
                                  apply_on_array=False),
                     Preprocessor('pick_channels', ch_names=ch_name,
                                  ordered=True),  # keep wanted channels
                     # clip all data within a given border
                     Preprocessor(scale,factor=1e6, apply_on_array=True),
                     Preprocessor(np.clip, a_min=crop_min,
                                  a_max=crop_max, apply_on_array=True),
                     Preprocessor('resample', sfreq=sfreq)]
    # TODO: shouldn't we add a bandstopfilter? though many before us has used this
    # Could add normalization here also
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    tuh_preproc = preprocess(
        concat_ds=concat_dataset,
        preprocessors=preprocessors,
        n_jobs=n_jobs,
        save_dir=save_dir,
        overwrite=True,
    )
    return tuh_preproc


def window_and_split(concat_ds: BaseConcatDataset, save_dir: str, overwrite=False,
                     window_size_samples=5000, n_jobs=1, channel_split_func=None,
                     save_dir_index=None) -> 'list[int]':
    if channel_split_func is None:
        channel_split_func = _make_adjacent_pairs
    if save_dir_index is None:
        save_dir = os.path.join(os.path.split(
            save_dir)[0], "split_indexes.pkl")
    # Drop too short samples
    concat_ds.set_description(
        {"n_samples": [ds.raw.n_times for ds in concat_ds.datasets]})  # type: ignore
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
    # Create save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not overwrite:
        _check_save_dir_empty(save_dir)
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
        delayed(_split_channels)(windows_ds, i,
                                 channel_split_dir, channel_split_func)
        for i, windows_ds in tqdm(enumerate(windows_ds.datasets), total=len(windows_ds.datasets))
    )
    print('Reloading serialized dataset')
    indexes = list(itertools.chain.from_iterable(indexes))  # type: ignore
    with open(save_dir_index, 'wb') as f:  # type: ignore
        pickle.dump(indexes, f)
    return indexes  # type: ignore


def _split_channels(windows_ds: WindowsDataset, record_index: int, save_dir: str, channel_split_func) -> 'list[int]':
    """Split single WindowsDataset into separate objects acording to channels picks

    Args:
        windows_ds (WindowsDataset): _description_
        record_index (int): _description_
        save_dir (str): _description_
        channel_split_func (_type_): _description_

    Returns:
        list[int]: _description_
    """
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
        # Create new WindowsDataset objects, which we will save
        ds = WindowsDataset(new_epochs, windows_ds.description)
        ds.window_kwargs = deepcopy(windows_ds.window_kwargs)  # type: ignore
        ds.set_description({"channels": channels})
        windows_ds_list.append(ds)

    concat_ds = BaseConcatDataset(windows_ds_list)
    concat_ds.save(save_dir, overwrite=True, offset=record_index*100)
    indexes = list(
        range(record_index*100, record_index*100+len(windows_ds_list)))
    return indexes


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
