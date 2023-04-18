import os
import pickle
import shutil
from copy import deepcopy

import braindecode.augmentation.functional
import numpy as np
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

import braindecode.datasets.tuh as tuh
import mne
from braindecode.datasets import BaseConcatDataset, WindowsDataset
from braindecode.datautil.serialization import _check_save_dir_empty, load_concat_dataset
from braindecode.preprocessing import create_fixed_length_windows, Preprocessor, preprocess, scale

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
excluded = sorted([
    "EEG EKG-REF", "EEG ROC-REF", "EEG EKG1-REF", "EEG C3P-REF", "EEG C4P-REF", "EEG LOC-REF", 'EEG EKG-LE',
    'PHOTIC PH', 'DC4-DC', 'DC3-DC', 'DC7-DC', 'DC2-DC', 'DC8-DC', 'DC6-DC', 'DC1-DC', 'DC5-DC', 'EMG-REF',
    'SUPPR', 'IBI', 'PHOTIC-REF', 'BURSTS', 'ECG EKG-REF', 'PULSE RATE', 'RESP ABDOMEN-REF','EEG RESP1-REF', 'EEG RESP2-REF'])


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


def preprocess_signals(concat_dataset: BaseConcatDataset, mapping, ch_naming, preproc_params, save_dir,
                       n_jobs, exclude_channels=[]):
    """
    renames channels to common naming, resamples all data to one frequency, sets common eeg_reference, applies
    bandpass filter and crops.
    :param concat_dataset: a TUH BaseConcatDataset of windowsDatasets
    :param mapping:
    :param n_jobs: number of available jobs for parallelization
    :return: preprocessed BaseConcatDataset
    """
    mne.set_log_level('ERROR')
    ch_naming = sorted(list(set(ch_naming) - set(exclude_channels)))
    # braindecode.augmentation.functional.channels_permute()
    preprocessors = [
        Preprocessor(custom_turn_off_log),  # turn off verbose
        # set common reference for all
        Preprocessor('set_eeg_reference',
                     ref_channels='average', ch_type='eeg'),
        # rename to common naming convention
        Preprocessor(rename_channels, mapping=mapping,
                     apply_on_array=False),
        Preprocessor('pick_channels', ch_names=ch_naming, ordered=False),  # keep wanted channels
        # Resample
        Preprocessor('resample', sfreq=preproc_params["s_freq"]),
        # rescale to microVolt (muV)
        Preprocessor(lambda data: np.multiply(data, preproc_params["scaling_factor"]), apply_on_array=True),
        # Bandpass filter
        Preprocessor('filter', l_freq=preproc_params["bandpass_lo"], h_freq=preproc_params["bandpass_hi"]),
        # clip all data within a given border
        Preprocessor(np.clip, a_min=preproc_params["crop_min"],
                     a_max=preproc_params["crop_max"], apply_on_array=True),
    ]
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


def window_ds(concat_ds: BaseConcatDataset,
              window_size_samples=5000, n_jobs=1, ) -> BaseConcatDataset:
    # Drop too short samples
    concat_ds.set_description(
        {"n_samples": [ds.raw.n_times for ds in concat_ds.datasets]})  # type: ignore
    keep = [n >= window_size_samples for n in concat_ds.description["n_samples"]]
    keep_indexes = [i for i, k in enumerate(keep) if k is True]
    concat_ds = concat_ds.split(by=keep_indexes)["0"]
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
    # store the number of windows required for loading later on
    n_windows = [len(ds) for ds in windows_ds.datasets]
    n_channels = [len(ds.windows.ch_names) for ds in windows_ds.datasets]
    windows_ds.set_description({
        "n_windows": n_windows,
        "n_channels": n_channels
    })

    return windows_ds


def split_by_channels(windowed_concat_ds: BaseConcatDataset, save_dir: str, n_jobs=1, channel_split_func=None,
                      overwrite=False, delete_step_1=False) -> 'list[tuple[int, int]]':
    if channel_split_func is None:
        channel_split_func = _make_adjacent_pairs
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
        _split_channels_parallel(windows_ds, i, save_dir, channel_split_func, delete_step_1)
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

    channel_selections = channel_split_func(epochs.ch_names)
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
        step_1_dir = windows_ds.windows.filename
        print(step_1_dir)

    indexes = list(
        range(record_index * 100, record_index * 100 + len(windows_ds_list)))
    return indexes, channel_n_windows


def _make_single_channels(ch_list: 'list[str]') -> 'list[list[str]]':
    return [[ch] for ch in ch_list]


def _make_unique_pair_combos(ch_list: 'list[str]') -> 'list[list[str]]':
    assert len(ch_list) > 1
    pairs = []
    for i, channel_i in enumerate(ch_list):
        for channel_j in ch_list[i + 1:]:
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
    for i in range(len(ch_list) // 2):
        pairs.append([ch_list[2 * i], ch_list[2 * i + 1]])
    if len(ch_list) % 2 == 1:
        pairs.append([ch_list[-1], ch_list[-2]])
    return pairs


def _make_overlapping_adjacent_pairs(ch_list: 'list[str]') -> 'list[list[str]]':
    assert len(ch_list) > 1
    pairs = []
    for i in range(len(ch_list) - 1):
        pairs.append([ch_list[i], ch_list[i + 1]])
    return pairs


string_to_channel_split_func = {
    "single": _make_single_channels,
    "unique_pairs": _make_unique_pair_combos,
    "all_pairs": _make_all_pair_combos,
    "adjacent_pairs": _make_adjacent_pairs,
    "overlapping_adjacent_pairs": _make_overlapping_adjacent_pairs,
}


def run_preprocess(config_path, to_numpy=False):
    # ------------------------ Read values from config file ------------------------
    with open(config_path, "r") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    source_ds = params["source_ds"]

    start_idx = params["start_idx"]
    stop_idx = params["stop_idx"]
    is_fine_tuning_ds = params["is_fine_tuning_ds"]
    if stop_idx <= 0:
        stop_idx = None

    assert source_ds in ['tuh_eeg_abnormal', 'tuh_eeg'], \
        f"{source_ds} is not a valid dataset option"
    read_cache = params["read_cache"]
    if (read_cache is None) or read_cache in [False, 'None']:
        read_cache = 'none'
    assert read_cache in ['none', 'raw', 'preproc', 'windows'], \
        f"{read_cache} is not a valid cache to read"

    local_load = params["local_load"]
    preproc_params = params["preprocess"]

    window_size = preproc_params["window_size"]
    channel_select_function = preproc_params["channel_select_function"]
    assert channel_select_function in string_to_channel_split_func.keys(), \
        f"{channel_select_function} is not a valid channel selection function"

    preproc_params = params["preprocess"]

    try:
        exclude_channels = preproc_params["exclude_channels"]
    except KeyError:
        exclude_channels = []

    # Read path info
    if local_load:
        paths = params['directory']['local']
    else:
        paths = params['directory']['disk']
    dataset_root = paths['dataset_root']
    cache_dir = paths['cache_dir']
    preproc_save_dir = paths['save_dir']
    split_save_dir = paths['save_dir_2']
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # -------------------------------- START PREPROC -------------------------------
    # Disable most MNE logging output which slows execution
    mne.set_log_level(verbose='ERROR')

    def _read_raw(start_idx=0, stop_idx=None):
        if source_ds == 'tuh_eeg_abnormal':
            dataset = tuh.TUHAbnormal(
                dataset_root,
                n_jobs=preproc_params['n_jobs'],
                target_name='pathological'
            )
        elif source_ds == 'tuh_eeg':
            dataset = tuh.TUH(dataset_root, n_jobs=preproc_params['n_jobs'])
        else:
            raise ValueError
        print(f'Loaded {len(dataset.datasets)} files.')
        if stop_idx is None:
            stop_idx = len(dataset.datasets)
        dataset = dataset.split(by=list(range(start_idx, stop_idx)))['0']
        # Cache pickle
        with open(os.path.join(cache_dir, 'raw_ds.pkl'), 'wb') as f:
            pickle.dump(dataset, f)
        # Next step:
        return _preproc_window(dataset=dataset)

    def _preproc_window(dataset=None, start_idx=0, stop_idx=None):
        """
        Window dataset
        """
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

        if not os.path.exists(split_save_dir):
            os.makedirs(split_save_dir)

        window_n_samples = int(preproc_params['window_size'] * preproc_params['s_freq'])
        print("Splitting dataset into windows:")
        windowed_ds = window_ds(dataset, window_size_samples=window_n_samples, n_jobs=preproc_params['n_jobs'])

        with open(os.path.join(cache_dir, 'windowed_ds.pkl'), 'wb') as f:
            pickle.dump(windowed_ds, f)

        # next step
        return _preproc_first(dataset=windowed_ds, start_idx=start_idx, stop_idx=stop_idx)

    def _preproc_first(dataset=None, start_idx=0, stop_idx=None):
        if dataset is None:
            print("Loading pickled windowed dataset...")
            with open(os.path.join(cache_dir, 'windowed_ds.pkl'), 'rb') as f:
                dataset = pickle.load(f)
                print('Done loading pickled windowed dataset.')
        if stop_idx is None:
            stop_idx = len(dataset.datasets)
        if len(dataset.datasets) > stop_idx - start_idx:
            dataset = dataset.split(by=range(start_idx, stop_idx))['0']
        # Create save_dir
        if not os.path.exists(preproc_save_dir):
            os.makedirs(preproc_save_dir)

        # with open(os.path.join(cache_dir, 'duration.pkl'), 'wb') as f:
        #     pickle.dump(dataset, f)
        print(f"Done. Kept {len(dataset.datasets)} files.")

        # Change channel names to common naming scheme for tuh_eeg
        common_naming, ch_mapping = create_channel_mapping()

        # Apply preprocessing step
        dataset = preprocess_signals(concat_dataset=dataset, mapping=ch_mapping,
                                     ch_naming=common_naming, preproc_params=preproc_params,
                                     save_dir=preproc_save_dir, n_jobs=preproc_params['n_jobs'],
                                     exclude_channels=exclude_channels)

        # Following pickle of dataset is disabled because of the
        # "cannot pickle '_io.BufferedReader' object" bug in braindecode

        # with open(os.path.join(cache_dir, 'preproc1_ds.pkl'), 'wb') as f:
        #     pickle.dump(dataset, f)

        # Next step, or return if fine-tuning set
        if is_fine_tuning_ds:
            return dataset
        else:
            return _preproc_split(dataset, start_idx=start_idx, stop_idx=stop_idx)

    def _preproc_split(dataset=None, start_idx=0, stop_idx=None):
        if dataset is None:
            print("Loading preprocessed dataset from file tree...")
            ids_to_load = list(range(start_idx, stop_idx))
            dataset = load_concat_dataset(preproc_save_dir, preload=False, n_jobs=preproc_params['n_jobs'],
                                          ids_to_load=ids_to_load)
            print('Done loading windowed dataset.')
        if stop_idx is None:
            stop_idx = len(dataset.datasets)
        if len(dataset.datasets) > stop_idx - start_idx:
            dataset = dataset.split(by=range(start_idx, stop_idx))['0']

        idx_list = split_by_channels(dataset, save_dir=split_save_dir, n_jobs=preproc_params['n_jobs'],
                                     channel_split_func=_make_adjacent_pairs, overwrite=True,
                                     delete_step_1=preproc_params["delete_step_1"])

        with open(os.path.join(cache_dir, 'split_idx_list.pkl'), 'wb') as f:
            pickle.dump(idx_list, f)
        return idx_list

    if read_cache == 'none':
        idx_list = _read_raw(start_idx=start_idx, stop_idx=stop_idx)
    elif read_cache == 'raw':
        idx_list = _preproc_window(start_idx=start_idx, stop_idx=stop_idx)
    elif read_cache == 'windows':
        idx_list = _preproc_first(start_idx=start_idx, stop_idx=stop_idx)
    elif read_cache == 'preproc':
        idx_list = _preproc_split(start_idx=start_idx, stop_idx=stop_idx)
    else:
        raise ValueError

    return idx_list
