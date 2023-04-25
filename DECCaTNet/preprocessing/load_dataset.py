import glob
import os

import numpy as np
import yaml
from tqdm import tqdm

import mne
from braindecode.datasets import BaseConcatDataset, tuh
from scipy.io import loadmat


def load_raw_tuh_eeg(ds_params, global_params) -> BaseConcatDataset:
    root_dir = ds_params['dataset_root']
    start_idx = ds_params['start_idx']
    stop_idx = ds_params['stop_idx']

    recording_ids = range(start_idx, stop_idx)
    dataset = tuh.TUH(root_dir, n_jobs=global_params['n_jobs'], recording_ids=recording_ids)
    return dataset


def load_raw_tuh_eeg_abnormal(ds_params, global_params) -> BaseConcatDataset:
    root_dir = ds_params['dataset_root']
    start_idx = ds_params['start_idx']
    stop_idx = ds_params['stop_idx']

    recording_ids = range(start_idx, stop_idx)
    dataset = tuh.TUHAbnormal(root_dir, recording_ids=recording_ids, n_jobs=global_params['n_jobs'],
                              target_name='pathological')
    return dataset


def load_raw_seed(ds_params, global_params, drop_non_eeg=True) -> BaseConcatDataset:
    root_dir = ds_params['dataset_root']
    start_idx = ds_params['start_idx']
    stop_idx = ds_params['stop_idx']

    file_paths = glob.glob(os.path.join(root_dir, '**/*.cnt'), recursive=True)
    file_paths = file_paths[start_idx:stop_idx]  # Keep only wanted part of dataset
    raws = []
    for file_path in tqdm(file_paths):
        raw = mne.io.read_raw_cnt(file_path, verbose='INFO')
        if drop_non_eeg:
            raw.drop_channels(['M1', 'M2', 'VEO', 'HEO'])
        raws.append(raw)

    # TODO: test seed load with disk,
    raw_1 = raws[0]
    print(raw_1.info)
    print(raw_1.ch_names)
    print(len(raw_1.ch_names))
    print(raw_1.info["dig"])

    dataset = BaseConcatDataset(raws)
    print(f'Loaded {len(dataset.datasets)} files.')
    return dataset


def load_raw_bciciv_1(ds_params, global_params) -> BaseConcatDataset:
    if ds_params["IS_FINE_TUNING_DS"]:
        print("BCICIV_1 is only implemented for pretraining")

    root_dir = ds_params['dataset_root']
    start_idx = ds_params['start_idx']
    stop_idx = ds_params['stop_idx']

    file_paths = glob.glob(os.path.join(root_dir, '**/*.mat'), recursive=True)
    file_paths = file_paths[start_idx:stop_idx]  # Keep only wanted part of dataset
    raws = []

    for file_path in tqdm(file_paths):
        mat = loadmat(file_path)

        for key in mat.keys():
            print(key)

        print(mat)

        data: np.ndarray = mat['cnt']
        print(data.shape)



load_func_dict = {
    'tuh_eeg': load_raw_tuh_eeg,
    'tuh_eeg_abnormal': load_raw_tuh_eeg_abnormal,
    'seed': load_raw_seed,
}

if __name__ == '__main__':
    # root_dir = 'D:/SEED/SEED_Multimodal/Chinese/01-EEG-raw'


    ds_params = {
        'dataset_root': 'C:/Users/oskar/repos/master-eeg-trans/datasets/BCICIV_1',
        'start_idx': 0,
        'stop_idx': 1,
        'IS_FINE_TUNING_DS': False,
    }
    global_params = {

    }

    ds = load_raw_bciciv_1(ds_params, global_params)
    print(ds)
