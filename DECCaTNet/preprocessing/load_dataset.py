import glob
import os

import yaml
from tqdm import tqdm

import mne
from braindecode.datasets import BaseConcatDataset, tuh


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

    # TODO: Remove print
    raw_1 = raws[0]
    print(raw_1.info)
    print(raw_1.ch_names)
    print(len(raw_1.ch_names))
    print(raw_1.info["dig"])

    dataset = BaseConcatDataset(raws)
    print(f'Loaded {len(dataset.datasets)} files.')
    return dataset


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


load_func_dict = {
    'tuh_eeg': load_raw_tuh_eeg,
    'tuh_eeg_abnormal': load_raw_tuh_eeg_abnormal,
    'seed': load_raw_seed,
}


if __name__ == '__main__':
    # dir = 'D:/SEED/SEED_Multimodal/Chinese/01-EEG-raw'
    config_path = r'C:\Users\oskar\repos\master-eeg-trans\DECCaTNet\DECCaTNet_model\configs\config_template.yaml'
    with open(config_path, 'r') as fid:
        params = yaml.safe_load(fid)
    concat_ds = load_raw_tuh_eeg(params["preprocess"]['tuh_eeg'], params['global'])
    print(concat_ds.datasets)
