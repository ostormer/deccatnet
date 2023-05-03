import gc
import itertools
import json
import os
import pickle
import shutil
from copy import deepcopy
from math import ceil

import numpy as np
from preprocessing.load_dataset import load_func_dict
from tqdm import tqdm

import mne
from braindecode.datasets import BaseConcatDataset, WindowsDataset, BaseDataset
from braindecode.datautil.serialization import _check_save_dir_empty, load_concat_dataset
from braindecode.preprocessing import create_fixed_length_windows, Preprocessor, preprocess

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

le_channels = sorted([
    'EEG 20-LE', 'EEG 21-LE', 'EEG 22-LE', 'EEG 23-LE', 'EEG 24-LE', 'EEG 25-LE', 'EEG 26-LE',
    'EEG 27-LE', 'EEG 28-LE', 'EEG 29-LE', 'EEG 30-LE', 'EEG 31-LE', 'EEG 32-LE', 'EEG A1-LE',
    'EEG A2-LE', 'EEG C3-LE', 'EEG C4-LE', 'EEG CZ-LE', 'EEG F3-LE', 'EEG F4-LE', 'EEG F7-LE',
    'EEG F8-LE',
    'EEG FP1-LE', 'EEG FP2-LE', 'EEG FZ-LE', 'EEG LUC-LE', 'EEG O1-LE', 'EEG O2-LE', 'EEG OZ-LE',
    'EEG P3-LE', 'EEG P4-LE', 'EEG PG1-LE', 'EEG PG2-LE', 'EEG PZ-LE', 'EEG RLC-LE', 'EEG SP1-LE',
    'EEG SP2-LE', 'EEG T1-LE', 'EEG T2-LE', 'EEG T3-LE', 'EEG T4-LE', 'EEG T5-LE', 'EEG T6-LE'])
ar_channels = sorted([
    'EEG 100-REF', 'EEG 101-REF', 'EEG 102-REF', 'EEG 103-REF', 'EEG 104-REF', 'EEG 105-REF',
    'EEG 106-REF', 'EEG 107-REF', 'EEG 108-REF', 'EEG 109-REF', 'EEG 110-REF', 'EEG 111-REF',
    'EEG 112-REF', 'EEG 113-REF', 'EEG 114-REF', 'EEG 115-REF', 'EEG 116-REF', 'EEG 117-REF',
    'EEG 118-REF', 'EEG 119-REF', 'EEG 120-REF', 'EEG 121-REF', 'EEG 122-REF', 'EEG 123-REF',
    'EEG 124-REF', 'EEG 125-REF', 'EEG 126-REF', 'EEG 127-REF', 'EEG 128-REF', 'EEG 20-REF',
    'EEG 21-REF', 'EEG 22-REF', 'EEG 23-REF', 'EEG 24-REF', 'EEG 25-REF', 'EEG 26-REF', 'EEG 27-REF',
    'EEG 28-REF', 'EEG 29-REF', 'EEG 30-REF', 'EEG 31-REF', 'EEG 32-REF', 'EEG 33-REF', 'EEG 34-REF',
    'EEG 35-REF', 'EEG 36-REF', 'EEG 37-REF', 'EEG 38-REF', 'EEG 39-REF', 'EEG 40-REF', 'EEG 41-REF',
    'EEG 42-REF', 'EEG 43-REF', 'EEG 44-REF', 'EEG 45-REF', 'EEG 46-REF', 'EEG 47-REF', 'EEG 48-REF',
    'EEG 49-REF', 'EEG 50-REF', 'EEG 51-REF', 'EEG 52-REF', 'EEG 53-REF', 'EEG 54-REF', 'EEG 55-REF',
    'EEG 56-REF', 'EEG 57-REF', 'EEG 58-REF', 'EEG 59-REF', 'EEG 60-REF', 'EEG 61-REF', 'EEG 62-REF',
    'EEG 63-REF', 'EEG 64-REF', 'EEG 65-REF', 'EEG 66-REF', 'EEG 67-REF', 'EEG 68-REF', 'EEG 69-REF',
    'EEG 70-REF', 'EEG 71-REF', 'EEG 72-REF', 'EEG 73-REF', 'EEG 74-REF', 'EEG 75-REF', 'EEG 76-REF',
    'EEG 77-REF', 'EEG 78-REF', 'EEG 79-REF', 'EEG 80-REF', 'EEG 81-REF', 'EEG 82-REF', 'EEG 83-REF',
    'EEG 84-REF', 'EEG 85-REF', 'EEG 86-REF', 'EEG 87-REF', 'EEG 88-REF', 'EEG 89-REF', 'EEG 90-REF',
    'EEG 91-REF', 'EEG 92-REF', 'EEG 93-REF', 'EEG 94-REF', 'EEG 95-REF', 'EEG 96-REF', 'EEG 97-REF',
    'EEG 98-REF', 'EEG 99-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG CZ-REF',
    'EEG F3-REF', 'EEG F4-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG FP1-REF', 'EEG FP2-REF',
    'EEG FZ-REF',
    'EEG LUC-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG OZ-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG PZ-REF',
    'EEG RLC-REF', 'EEG SP1-REF', 'EEG SP2-REF', 'EEG T1-REF',
    'EEG T2-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF'])


# excluded = sorted([
#     "EEG EKG-REF", "EEG ROC-REF", "EEG EKG1-REF", "EEG C3P-REF", "EEG C4P-REF", "EEG LOC-REF", 'EEG EKG-LE',
#     'PHOTIC PH', 'DC4-DC', 'DC3-DC', 'DC7-DC', 'DC2-DC', 'DC8-DC', 'DC6-DC', 'DC1-DC', 'DC5-DC', 'EMG-REF',
#     'SUPPR', 'IBI', 'PHOTIC-REF', 'BURSTS', 'ECG EKG-REF', 'PULSE RATE', 'RESP ABDOMEN-REF', 'EEG RESP1-REF',
#     'EEG RESP2-REF'])


def select_duration(concat_ds: BaseConcatDataset, t_min=0, t_max: int = None):
    """
    Selects all recordings which fulfills: t_min < duration < t_max, data is not loaded here, only metadata
    :param concat_ds: dataset
    :param t_min: minimum length of recordings
    :param t_max: maximum length of recordings
    :return: satisfactory recordings
    """
    if t_max is None:
        t_max = np.Inf
    idx = []
    for i, ds in enumerate(concat_ds.datasets):
        len_session = ds.raw.tmax - ds.raw.tmin  # type: ignore
        if t_min < len_session < t_max:
            idx.append(i)
    good_recordings = [concat_ds.datasets[i] for i in idx]
    return BaseConcatDataset(good_recordings)


def rename_channels(raw, mapping):
    """
    Renames all channels such that the naming conventions is similar for all recordings
    This will  work as a custom preprocessor functions form  braindecode preprocessors
    :param raw: mne raw object
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


def create_channel_mapping():
    # create mapping from channel names to channel
    ar_common_naming = sorted(list(set([x.split('-')[0] for x in ar_channels])))
    le_common_naming = sorted(list(set([x.split('-')[0] for x in le_channels])))
    common_naming = sorted(list(set(ar_common_naming + le_common_naming)))
    # create dictionaries with key ar or le channel name and send to common name
    ar_to_common = {ar_ref: common for ar_ref, common in zip(ar_channels, ar_common_naming)}
    le_to_common = {le_ref: common for le_ref, common in zip(le_channels, le_common_naming)}
    ch_mapping = {'ar': ar_to_common, 'le': le_to_common}

    return common_naming, ch_mapping


def custom_turn_off_log(raw, verbose='ERROR'):
    mne.set_log_level(verbose)
    return raw


def window_ds(concat_ds: BaseConcatDataset, preproc_params, global_params) -> BaseConcatDataset:
    n_jobs = global_params['n_jobs']
    window_size = global_params['window_size']
    save_dir = preproc_params['preproc_save_dir']
    # Drop too short recordings and uninteresting channels
    keep_ds = []
    print('Dropping short recordings and excluded channels:')
    try:
        exclude_ch = preproc_params['exclude_channels']
    except IndexError:
        exclude_ch = []
    for ds in tqdm(concat_ds.datasets):
        if ds.raw.n_times * ds.raw.info['sfreq'] >= window_size:
            keep_ds.append(ds)
        ds.raw.drop_channels(exclude_ch, on_missing='ignore')
    print(f'Kept  {len(keep_ds)} recordings')
    concat_ds = BaseConcatDataset(keep_ds)

    print("Rescaling to microVolts...")
    preprocessors = [
        Preprocessor(custom_turn_off_log),  # turn off verbose
        Preprocessor(lambda data: np.multiply(data, preproc_params["scaling_factor"]), apply_on_array=True),
    ]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    preprocess(concat_ds=concat_ds,  # preprocess is in place, doesnt return anything because overwrtie=True
               preprocessors=preprocessors,
               n_jobs=n_jobs,
               save_dir=save_dir,
               overwrite=True,
               )
    print("Done rescaling to microVolts")

    if preproc_params['reject_high_threshold'] is None:
        reject_dict = None
    else:
        reject_dict = dict(eeg=preproc_params['reject_high_threshold'])
    if preproc_params['reject_flat_threshold'] is None:
        flat_dict = None
    else:
        flat_dict = dict(eeg=preproc_params['reject_flat_threshold'])
    print('Splitting dataset into windows:')
    windows_ds = create_fixed_length_windows(
        concat_ds,
        n_jobs=n_jobs,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_seconds=window_size,
        drop_last_window=False,
        drop_bad_windows=True,
        reject=reject_dict,  # Peak-to-peak high rejection threshold within each window
        flat=flat_dict,  # Peak-to peak low rejection threshold
        verbose='ERROR'
    )
    # Trying to free up memory
    for ds in concat_ds.datasets:
        del ds.raw
        del ds
    del concat_ds
    gc.collect()

    keep_ds = []
    print('Dropping all recordings with 0 good windows...')
    for ds in tqdm(windows_ds.datasets):
        if len(ds.windows) > 0:
            keep_ds.append(ds)
    print(f'Kept {len(keep_ds)} recordings')
    windows_ds = BaseConcatDataset(keep_ds)

    n_windows = [len(ds) for ds in windows_ds.datasets]
    n_channels = [len(ds.windows.ch_names) for ds in windows_ds.datasets]
    windows_ds.set_description({
        "n_windows": n_windows,
        "n_channels": n_channels
    })
    return windows_ds


def preprocess_signals(concat_dataset: BaseConcatDataset, mapping, ch_naming, preproc_params, save_dir,
                       n_jobs, s_freq, exclude_channels=None):
    """
    renames channels to common naming, resamples all data to one frequency, sets common eeg_reference, applies
    bandpass filter and crops.
    :param concat_dataset: a TUH BaseConcatDataset of windowsDatasets
    :param mapping:
    :param n_jobs: number of available jobs for parallelization
    :return: preprocessed BaseConcatDataset
    """
    if exclude_channels is None:
        exclude_channels = []
    mne.set_log_level('ERROR')
    ch_naming = sorted(list(set(ch_naming) - set(exclude_channels)))

    preprocessors = [
        Preprocessor(custom_turn_off_log),  # turn off verbose
        # set common reference for all
        Preprocessor('set_eeg_reference',
                     ref_channels='average', ch_type='eeg'),
        # rename to common naming convention
        # Preprocessor(rename_channels, mapping=mapping, apply_on_array=False),
        Preprocessor('pick_channels', ch_names=ch_naming, ordered=False),  # keep wanted channels
        # Resample
        Preprocessor('resample', sfreq=s_freq),
        # Bandpass filter
        Preprocessor('filter', l_freq=preproc_params["bandpass_lo"], h_freq=preproc_params["bandpass_hi"]),
    ]
    #if preproc_params['IS_FINE_TUNING_DS']:
    #    preprocessors.append(Preprocessor('equalize_channels', copy=False))

    # Could add normalization here also

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    preprocess(concat_ds=concat_dataset,  # preprocess is in place, doesnt return anything because overwrtie=True
               preprocessors=preprocessors,
               n_jobs=n_jobs,
               save_dir=save_dir,
               overwrite=True,
               )
    # check_windows(concat_dataset)
    # Divide data by trial: Check sampling frequency

    return concat_dataset


def check_windows(concat_dataset):
    metadata = concat_dataset.datasets[0].windows.metadata.iloc[[0]]
    window_len = metadata['i_stop_in_trial'] - metadata['i_start_in_trial']

    if not all([(get_window_len(ds, window_len.to_numpy()[0]) == window_len.to_numpy()[0]).all() for ds in
                concat_dataset.datasets]):
        raise ValueError("Not match length of windows.")

    sfreq = concat_dataset.datasets[0].windows.info['sfreq']
    if not all([ds.windows.info['sfreq'] == sfreq for ds in concat_dataset.datasets]):
        print([(ds.windows.info['sfreq'], ds.info) for ds in concat_dataset.datasets if
               ds.windows.info['sfreq'] != sfreq])
        raise ValueError(f"Not match sampling rate. {sfreq}")


def get_window_len(ds, window_len):
    diff = ds.windows.metadata['i_stop_in_trial'] - ds.windows.metadata['i_start_in_trial']
    if diff.to_numpy()[0] > window_len:
        print(f'window_len {diff.to_numpy()[0]} target: {window_len} and info: {ds.windows.info}')
    return diff.to_numpy()


def split_by_channels(windowed_concat_ds: BaseConcatDataset, save_dir: str, n_channels, channel_split_func=None,
                      overwrite=False, delete_step_1=False) -> 'list[tuple[int, int]]':
    if channel_split_func is None:
        channel_split_func = _make_adjacent_groups
    # Create save_dir for channel split dataset
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

    # Channel splitting
    print('Splitting recordings into separate channels')

    # Since the ds is now windowed, and we cannot parallelize epochs object because of the
    # "cannot pickle '_io.BufferedReader' object" bug in braindecode,
    # we do not parallelize this part

    # idx_n_windows = Parallel(n_jobs=n_jobs)(
    #     delayed(_split_channels_parallel)(windows_ds, i, save_dir, channel_split_func, delete_step_1)
    #     for i, windows_ds in tqdm(enumerate(windowed_concat_ds.datasets), total=len(windowed_concat_ds.datasets))
    # )

    # Not parallelized equivalent of the above
    idx_n_windows = [
        _split_channels_parallel(windows_ds, i, save_dir, n_channels, channel_split_func, delete_step_1)
        for i, windows_ds in tqdm(enumerate(windowed_concat_ds.datasets), total=len(windowed_concat_ds.datasets))
    ]

    print('Creating idx list')
    # make list with an element for each window in the entire dataset,
    # pointing at file and window number.
    idx_list = []
    for pair in tqdm(idx_n_windows):
        # For all recordings in the dataset
        dir_names, n_windows = pair
        for d, w in zip(dir_names, n_windows):
            # For all windows_datasets originating from one recording
            for window_index in range(w):
                idx_list.append((d, window_index))

    return idx_list


def _split_channels_parallel(
        windows_ds: WindowsDataset,
        record_index: int,
        save_dir: str,
        n_channels: int,
        channel_split_func,
        delete_step_1
) -> 'tuple[list[int], list[int]]':
    """Split single WindowsDataset into separate objects according to channels picks

    Args:
        windows_ds (WindowsDataset): _description_
        record_index (int): _description_
        save_dir (str): _description_
        channel_split_func (_type_): _description_

    Returns:
        list[int]: _description_
    """
    mne.set_log_level(verbose='ERROR')
    epochs = windows_ds.windows

    channel_selections = channel_split_func(epochs.ch_names, n_channels=n_channels)
    windows_ds_list = []
    channel_n_windows = []
    epochs.load_data()
    for channels in channel_selections:
        new_epochs = deepcopy(epochs).pick(channels)
        new_epochs.drop_bad()

        # Create new WindowsDataset objects, which we will save
        ds = WindowsDataset(new_epochs, windows_ds.description)
        channel_n_windows.append(len(ds))
        ds.window_kwargs = deepcopy(windows_ds.window_kwargs)  # type: ignore
        ds.set_description({"channels": channels})
        windows_ds_list.append(ds)
    concat_ds = BaseConcatDataset(windows_ds_list)
    concat_ds.save(save_dir, overwrite=True, offset=record_index * 100)

    if delete_step_1:
        # The dir containing one preprocessed WindowsDataset and acompanying json files
        step_1_dir = os.path.join(*os.path.split(windows_ds.windows.filename)[:-1])
        del windows_ds
        gc.collect()
        shutil.rmtree(step_1_dir)

    indexes = list(
        range(record_index * 100, record_index * 100 + len(windows_ds_list)))
    return indexes, channel_n_windows


def _make_all_combinations(ch_list: 'list[str]', n_channels) -> 'list[list[str]]':
    assert len(ch_list) >= n_channels
    groups = []
    for g in itertools.combinations(ch_list, n_channels):
        groups.append(list(g))
    return groups


def _make_all_permutations(ch_list: 'list[str]', n_channels) -> 'list[list[str]]':
    assert len(ch_list) >= n_channels
    groups = []
    for g in itertools.permutations(ch_list, n_channels):
        groups.append(list(g))
    return groups


def _make_adjacent_groups(ch_list: 'list[str]', n_channels) -> 'list[list[str]]':
    assert len(ch_list) >= n_channels
    n_groups = ceil(len(ch_list) / n_channels)
    last_start_idx = len(ch_list) - n_channels  # First channel in final group
    # evenly spaced group starts, so overlap is distributed evenly
    start_idx = np.round(np.linspace(0, last_start_idx, n_groups)).astype('int')
    groups = []
    for i in start_idx:
        groups.append(ch_list[i:i + n_channels])
    return groups


def _make_overlapping_adjacent_groups(ch_list: 'list[str]', n_channels) -> 'list[list[str]]':
    assert len(ch_list) > n_channels
    groups = []
    for i in range(len(ch_list) - n_channels + 1):
        groups.append(ch_list[i:i + n_channels])
    return groups


string_to_channel_split_func = {
    "permutations": _make_all_permutations,
    "combinations": _make_all_combinations,
    "adjacent_groups": _make_adjacent_groups,
    "overlapping_adjacent_grooups": _make_overlapping_adjacent_groups,
}


def _read_raw(ds_params, global_params):
    """
    Wrapper function that selects the right loader function according to the dataset, then runs it.
    """
    ds_name = ds_params['ds_name']
    cache_dir = ds_params['cache_dir']

    ds_load_func = load_func_dict[ds_name]
    dataset = ds_load_func(ds_params, global_params)
    print(f'Loaded {len(dataset.datasets)} files.')

    # Cache pickle
    with open(os.path.join(cache_dir, 'raw_ds.pkl'), 'wb') as f:
        pickle.dump(dataset, f)
    # Next step:
    return _preproc_window(ds_params, global_params, dataset=dataset)


def _preproc_window(ds_params, global_params, dataset=None):
    """
    Wrapper function to load pickled BaseConcatDataset if needed, then window it
    """
    start_idx = ds_params['start_idx']
    stop_idx = ds_params['stop_idx']
    cache_dir = ds_params['cache_dir']
    window_size = global_params['window_size']
    if dataset is None:
        print("Loading pickled dataset from file")
        with open(os.path.join(cache_dir, 'raw_ds.pkl'), 'rb') as f:
            dataset = pickle.load(f)
            print('Done loading pickled raw dataset.')
        if stop_idx is None:
            stop_idx = len(dataset.datasets)
        if len(dataset.datasets) > stop_idx - start_idx:
            dataset = dataset.split(by=range(start_idx, stop_idx))['0']

    # Select by duration
    print(f'Start selecting samples with duration over {window_size} sec')
    dataset = select_duration(dataset, t_min=window_size, t_max=None)
    print('Windowing dataset...')
    windowed_ds = window_ds(dataset, preproc_params=ds_params, global_params=global_params)

    with open(os.path.join(cache_dir, 'windowed_ds.pkl'), 'wb') as f:
        pickle.dump(windowed_ds, f)
    # next step
    return _preproc_first(ds_params, global_params, dataset=windowed_ds)


def _preproc_first(ds_params, global_params, dataset=None):
    """
    Wrapper function to load pickled WindowsDataset if needed, then run preprocessing pipeline on it
    """
    start_idx = ds_params['start_idx']
    stop_idx = ds_params['stop_idx']
    preproc_save_dir = ds_params['preproc_save_dir']
    cache_dir = ds_params['cache_dir']
    is_fine_tuning_ds = ds_params['IS_FINE_TUNING_DS']
    try:
        exclude_channels = ds_params["exclude_channels"]
    except KeyError:
        exclude_channels = []

    if dataset is None:
        print("Loading pickled windowed dataset...")
        with open(os.path.join(cache_dir, 'windowed_ds.pkl'), 'rb') as f:
            dataset = pickle.load(f)
            print('Done loading pickled windowed dataset.')
    if stop_idx is None:
        stop_idx = len(dataset.datasets)
    if len(dataset.datasets) > stop_idx - start_idx:
        dataset = dataset.split(by=range(start_idx, stop_idx))['0']

    print("Keeping values from ")
    # Change channel names to common naming scheme for tuh_eeg
    common_naming, ch_mapping = create_channel_mapping()
    # Create preproc_save_dir
    if not os.path.exists(preproc_save_dir):
        os.makedirs(preproc_save_dir)
    else:
        # Delete all other folders
        keep_dirs = set([str(i) for i in range(start_idx, stop_idx)])
        print(start_idx, stop_idx)
        dirs_from_previous_step = set(os.listdir(preproc_save_dir))
        dirs_to_delete = sorted(list(dirs_from_previous_step - keep_dirs))
        print("Deleting following dirs as they are from before the preprocessing overwrote the others")
        print(dirs_to_delete)
        for ds_dir in dirs_to_delete:
            try:
                shutil.rmtree(os.path.join(preproc_save_dir, ds_dir))
            except OSError:
                os.remove(os.path.join(preproc_save_dir, ds_dir))


    # Apply preprocessing step
    dataset = preprocess_signals(concat_dataset=dataset, mapping=ch_mapping,
                                 ch_naming=common_naming, preproc_params=ds_params,
                                 save_dir=preproc_save_dir, n_jobs=global_params['n_jobs'],
                                 exclude_channels=exclude_channels, s_freq=global_params['s_freq'])
    # Next step, or return if fine-tuning set
    if is_fine_tuning_ds and not global_params['HYPER_SEARCH']:
        return dataset
    elif is_fine_tuning_ds:
        return None
    else:
        return _preproc_split(ds_params, global_params, dataset)


def _save_fine_tuning_ds(ds_params, global_params, orig_dataset=None):
    """
    saves a fine_tuning dataset in the same way as a pre_training dataset to make it pickleable
    :param ds_params: dataset parameters
    :param global_params: global parameters
    :param dataset: fine_tuning ds which needs to be saved
    :return: a finetuning PathDataset
    """
    # first ensure that everything is on the correct order
    # print(f'started to equalize channels')
    # windows = [window.windows for window in orig_dataset.datasets]
    # windows = mne.equalize_channels(windows)

    save_dir = ds_params['split_save_dir']

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    idx = []
    for i, window_ds in enumerate(orig_dataset.datasets):
        # print(window_ds.description)
        for i_window in range(len(window_ds)):
            idx.append((i, i_window))

    print(f'saving BaseConcatDataset to {save_dir}, with the following idx: {idx}')
    dataset = BaseConcatDataset(orig_dataset.datasets)
    dataset.save(save_dir, overwrite=True)
    return idx, save_dir, ds_params


def _preproc_split(ds_params, global_params, dataset=None):
    """
    Wrapper function to load pickled dataset if needed, and then run channel split
    """
    start_idx = ds_params['start_idx']
    stop_idx = ds_params['stop_idx']
    preproc_save_dir = ds_params['preproc_save_dir']
    split_save_dir = ds_params['split_save_dir']
    cache_dir = ds_params['cache_dir']

    if dataset is None:
        print("Loading preprocessed dataset from file tree...")
        if start_idx == 0 and stop_idx is None:
            dataset = load_concat_dataset(preproc_save_dir, preload=False, n_jobs=global_params['n_jobs'])
        else:
            ids_to_load = list(range(start_idx, stop_idx))
            dataset = load_concat_dataset(preproc_save_dir, preload=False, n_jobs=global_params['n_jobs'],
                                        ids_to_load=ids_to_load)
        print('Done loading preprocessed dataset.')
    if stop_idx is None:
        stop_idx = len(dataset.datasets)
    if len(dataset.datasets) > stop_idx - start_idx:
        dataset = dataset.split(by=list(range(start_idx, stop_idx)))['0']

    if ds_params["IS_FINE_TUNING_DS"]:  # Return loaded and possibly cut dataset
        return dataset

    if not os.path.exists(split_save_dir):
        os.makedirs(split_save_dir)

    idx_list = split_by_channels(dataset, save_dir=split_save_dir, n_channels=global_params['n_channels'],
                                 channel_split_func=_make_adjacent_groups, overwrite=True,
                                 delete_step_1=ds_params["DELETE_STEP_1"])

    with open(os.path.join(cache_dir, 'split_idx_list.pkl'), 'wb') as f:
        pickle.dump(idx_list, f)

    return idx_list


def run_preprocess(params_all, global_params, fine_tuning=False):
    """
    Run preprocessing according to parameters in config file

    Sets up config dicts and runs preprocessing accordingly
    """
    # ------------------------ Perform preprocessing ------------------------
    if fine_tuning:
        params_all['preprocess'][params_all['fine_tuning']['ds_name']] = params_all['preprocess']
        datasets = [params_all['fine_tuning']['ds_name']]
    else:
        datasets = global_params['datasets']

    preproc_datasets = []

    for dataset_name in datasets:
        print(f"========= Beginning preprocessing pipeline for {dataset_name} =========")
        ds_params = params_all['preprocess'][dataset_name]

        assert dataset_name in load_func_dict.keys(), \
            f"{dataset_name} is not an implemented dataset name"
        read_cache = ds_params["read_cache"]
        if (read_cache is None) or read_cache in [False, 'None']:
            read_cache = 'none'
        assert read_cache in ['none', 'raw', 'preproc', 'windows'], \
            f"{read_cache} is not a valid cache to read"

        channel_select_function = global_params["channel_select_function"]
        assert channel_select_function in string_to_channel_split_func.keys(), \
            f"{channel_select_function} is not a valid channel selection function"

        ds_params['cache_dir'] = os.path.join(ds_params['preprocess_root'], 'pickles')
        ds_params['preproc_save_dir'] = os.path.join(ds_params['preprocess_root'], 'first_preproc')
        ds_params['split_save_dir'] = os.path.join(ds_params['preprocess_root'], 'split')

        if not os.path.exists(ds_params["cache_dir"]):
            os.makedirs(ds_params["cache_dir"])

        # -------------------------------- START PREPROC -------------------------------
        # Disable most MNE logging output which slows execution
        mne.set_log_level(verbose='ERROR')
        if read_cache == 'none':
            idx_list = _read_raw(ds_params, global_params)
        elif read_cache == 'raw':
            idx_list = _preproc_window(ds_params, global_params)
        elif read_cache == 'windows':
            idx_list = _preproc_first(ds_params, global_params)
        elif read_cache == 'preproc':
            idx_list = _preproc_split(ds_params, global_params)
        else:
            raise ValueError
        preproc_datasets.append(idx_list)

    return preproc_datasets
