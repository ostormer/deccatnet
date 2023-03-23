import os
import pickle
import yaml

import braindecode.datasets.tuh as tuh
import mne
from braindecode.datautil.serialization import load_concat_dataset

from .preprocess import string_to_channel_split_func, window_and_split, select_duration, \
    first_preprocess_step, create_channel_mapping

def run_preprocess(config_path, start_idx=0, stop_idx=None):
    # ------------------------ Read values from config file ------------------------
    with open(config_path, "r") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    read_cache = params["read_cache"]
    if (read_cache is None) or read_cache in [False, 'None']:
        read_cache = 'none'

    assert read_cache in ['none', 'raw', 'preproc', 'windows', 'split'], \
        f"{read_cache} is not a valid cache to read"

    source_ds = params["source_ds"]
    assert source_ds in ['tuh_eeg_abnormal', 'tuh_eeg'], \
        f"{source_ds} is not a valid dataset option"

    local_load = params["local_load"]
    preproc_params = params["preprocess"]

    window_size = preproc_params["window_size"]
    channel_select_function = preproc_params["channel_select_function"]
    assert channel_select_function in string_to_channel_split_func.keys(), \
        f"{channel_select_function} is not a valid channel selection function"

    preproc_params = params["preprocess"]

    # Read path info
    if local_load:
        paths = params['directory']['local']
    else:
        paths = params['directory']['disk']
    dataset_root = paths['dataset_root']
    cache_dir = paths['cache_dir']
    save_dir = paths['save_dir']
    save_dir_2 = paths['save_dir_2']
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # -------------------------------- START PREPROC -------------------------------
    # Disable most MNE logging output which slows execution
    mne.set_log_level(verbose='ERROR')

    def _read_raw(start_idx=0, stop_idx=None):
        if source_ds == 'tuh_eeg_abnormal':
            dataset = tuh.TUHAbnormal(dataset_root, n_jobs=preproc_params['n_jobs'])
        elif source_ds == 'tuh_eeg':
            dataset = tuh.TUH(dataset_root, n_jobs=preproc_params['n_jobs'])
        else:
            raise ValueError
        print(f'Loaded {len(dataset.datasets)} files.')
        if stop_idx is None:
            stop_idx = len(dataset.datasets)
        dataset = dataset.split(by=range(start_idx, stop_idx))['0']
        # Cache pickle
        with open(os.path.join(cache_dir, 'raw.pkl'), 'wb') as f:
            pickle.dump(dataset, f)
        # Next step:
        return _preproc_first(dataset=dataset)

    def _preproc_first(dataset=None, start_idx=0, stop_idx=None):
        if dataset is None:
            print("Loading pickled raw dataset...")
            with open(os.path.join(cache_dir, 'raw.pkl'), 'rb') as f:
                dataset = pickle.load(f)
                if stop_idx is None:
                    stop_idx = len(dataset.datasets)
                dataset = dataset.split(by=range(start_idx, stop_idx))['0']
                print('Done loading pickled raw dataset.')
        if stop_idx is None:
            stop_idx = len(dataset.datasets)
        # Create save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Select by duration
        print(f'Start selecting samples with duration over {window_size} sec')
        dataset = select_duration(dataset, t_min=window_size, t_max=None)

        with open(os.path.join(cache_dir, 'duration.pkl'), 'wb') as f:
            pickle.dump(dataset, f)

        print(f"Done. Kept {len(dataset.datasets)} files.")

        # Change channel names to common naming scheme
        # For tuh_eeg
        common_naming, ch_mapping = create_channel_mapping()

        # Apply preprocessing step
        dataset = first_preprocess_step(concat_dataset=dataset, mapping=ch_mapping,
                                        ch_naming=common_naming, crop_min=preproc_params['crop_min'],
                                        crop_max=preproc_params['crop_max'], sfreq=preproc_params['s_freq'],
                                        save_dir=save_dir, n_jobs=preproc_params['n_jobs'])
        with open(os.path.join(cache_dir, 'preproc1.pkl'), 'wb') as f:
            pickle.dump(dataset, f)
        # next step
        return _preproc_window(dataset, start_idx=start_idx, stop_idx=stop_idx)

    def _preproc_window(dataset=None, start_idx=0, stop_idx=None):
        if dataset is None:
            print("Loading pickled raw dataset...")
            with open(os.path.join(cache_dir, 'preproc1.pkl'), 'rb') as f:
                dataset = pickle.load(f)
                if stop_idx is None:
                    stop_idx = len(dataset.datasets)
                dataset = dataset.split(by=range(start_idx, stop_idx))['0']
                print('Done loading pickled raw dataset.')

        if not os.path.exists(save_dir_2):
            os.makedirs(save_dir_2)

        window_n_samples = preproc_params['window_size'] * preproc_params['s_freq']
        print("Splitting dataset into windows:")
        idx_dict = window_and_split(dataset, save_dir=save_dir_2, overwrite=True,
                                    window_size_samples=window_n_samples, n_jobs=preproc_params['n_jobs'])

        with open(os.path.join(cache_dir, 'windowed_ids.pkl'), 'wb') as f:
            pickle.dump(idx_dict, f)
        return idx_dict



    if read_cache == 'none':
        idx_dict = _read_raw(start_idx=start_idx, stop_idx=stop_idx)
    elif read_cache == 'raw':
        idx_dict = _preproc_first(start_idx=start_idx, stop_idx=stop_idx)
    elif read_cache == 'preproc':
        idx_dict = _preproc_window(start_idx=start_idx, stop_idx=stop_idx)
    else:
        raise ValueError

    return idx_dict



