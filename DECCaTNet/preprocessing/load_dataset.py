import glob
import os

import numpy
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

import mne
from braindecode.datasets import BaseDataset, BaseConcatDataset, tuh
from braindecode.datautil import load_concat_dataset

excluded_tuh = sorted([
    "EEG EKG-REF", "EEG ROC-REF", "EEG EKG1-REF", "EEG C3P-REF", "EEG C4P-REF", "EEG LOC-REF", 'EEG EKG-LE',
    'PHOTIC PH', 'DC4-DC', 'DC3-DC', 'DC7-DC', 'DC2-DC', 'DC8-DC', 'DC6-DC', 'DC1-DC', 'DC5-DC', 'EMG-REF',
    'SUPPR', 'IBI', 'PHOTIC-REF', 'BURSTS', 'ECG EKG-REF', 'PULSE RATE', 'RESP ABDOMEN-REF', 'EEG RESP1-REF',
    'EEG RESP2-REF'])

def load_raw_tuh_eeg(ds_params, global_params) -> BaseConcatDataset:
    root_dir = ds_params['dataset_root']
    start_idx = ds_params['start_idx']
    stop_idx = ds_params['stop_idx']

    try:
        recording_ids = range(start_idx, stop_idx)
    except:
        recording_ids = None

    dataset = tuh.TUH(root_dir, n_jobs=global_params['n_jobs'], recording_ids=recording_ids)
    for i,ds in enumerate(dataset.datasets):
        for channel in excluded_tuh:
            try:
                ds.raw.drop_channels(channel)
            except:
                pass
                #print(f'ValueError: Channel(s){channel} most likely not found in raw file number: {i}')
    return dataset

def load_raw_tuh_eeg_abnormal(ds_params, global_params) -> BaseConcatDataset:
    root_dir = ds_params['dataset_root']
    start_idx = ds_params['start_idx']
    stop_idx = ds_params['stop_idx']

    try:
        recording_ids = range(start_idx, stop_idx)
    except:
        recording_ids = None

    dataset = tuh.TUHAbnormal(root_dir, recording_ids=recording_ids, n_jobs=global_params['n_jobs'],
                              target_name='pathological')
    for i,ds in enumerate(dataset.datasets):
        for channel in excluded_tuh:
            try:
                ds.raw.drop_channels(channel)
            except:
                pass
                #print(f'ValueError: Channel(s){channel} most likely not found in raw file number: {i}')
    return dataset


def load_raw_seed(ds_params, global_params, drop_non_eeg=True) -> BaseConcatDataset:
    root_dir = ds_params['dataset_root']
    start_idx = ds_params['start_idx']
    stop_idx = ds_params['stop_idx']

    file_paths = glob.glob(os.path.join(root_dir, '**/*.cnt'), recursive=True)
    file_paths = file_paths[start_idx:stop_idx]  # Keep only wanted part of dataset
    base_datasets = []
    for file_path in tqdm(file_paths):
        raw = mne.io.read_raw_cnt(file_path, verbose='INFO', preload=False)
        if drop_non_eeg:
            raw.drop_channels(['M1', 'M2', 'VEO', 'HEO'])
        description = {"file_path": file_path}
        ds = BaseDataset(raw, description=description, target_name=None)
        base_datasets.append(ds)

    dataset = BaseConcatDataset(base_datasets)
    return dataset


def load_raw_bciciv_1(ds_params, global_params) -> BaseConcatDataset:
    if ds_params["IS_FINE_TUNING_DS"]:
        print("BCICIV_1 is only implemented for pretraining")

    root_dir = ds_params['dataset_root']
    start_idx = ds_params['start_idx']
    stop_idx = ds_params['stop_idx']

    file_paths = glob.glob(os.path.join(root_dir, '**/*.mat'), recursive=True)
    file_paths = file_paths[start_idx:stop_idx]  # Keep only wanted part of dataset

    base_datasets = []
    for file_path in tqdm(file_paths):
        mat = loadmat(file_path)
        raw_nfo = mat['nfo']
        # Create info object
        sfreq: int = raw_nfo['fs'][0][0][0][0]  # WTF why is it packed so deep
        ch_names = [ch[0] for ch in raw_nfo['clab'][0][0][0]]
        info = mne.create_info(
            ch_names,
            sfreq,
            ch_types=["eeg"] * len(ch_names),
            verbose="ERROR"
        )
        info['description'] = f'File: {file_path}'

        # Electrode position handling
        # x_pos = [pos[0] for pos in raw_nfo['xpos'][0][0]]
        # y_pos = [pos[0] for pos in raw_nfo['ypos'][0][0]]
        # plt.plot(x_pos, y_pos, '.')
        # plt.show()
        # points_3d = []
        # for flat_x, flat_y in zip(x_pos, y_pos):
        #     # https://en.wikipedia.org/wiki/Azimuthal_equidistant_projection
        #     theta = 0  #  Just give up on this for a while, its not needed now
        # Prepare data
        data: np.ndarray = mat['cnt']
        data = data.transpose().astype(numpy.float32)
        raw = mne.io.RawArray(data, info, verbose='ERROR')  # Create raw object

        description = {"file_path": file_path}
        ds = BaseDataset(raw, description=description, target_name=None)
        base_datasets.append(ds)
    concat_dataset = BaseConcatDataset(base_datasets)
    print("Serializing braindecode dataset to free up memory...")
    concat_dataset.save(ds_params["preproc_save_dir"], overwrite=True)
    concat_dataset = load_concat_dataset(path=ds_params["preproc_save_dir"], preload=False, n_jobs=global_params["n_jobs"])

    return concat_dataset


load_func_dict = {
    'tuh_eeg': load_raw_tuh_eeg,
    'tuh_eeg_abnormal': load_raw_tuh_eeg_abnormal,
    'seed': load_raw_seed,
    'bciciv_1': load_raw_bciciv_1,
}

if __name__ == '__main__':
    # root_dir = 'D:/SEED/SEED_Multimodal/Chinese/01-EEG-raw'

    ds_params = {
        'dataset_root': 'C:/Users/oskar/repos/master-eeg-trans/datasets/BCICIV_1',
        'start_idx': 0,
        'stop_idx': None,
        'IS_FINE_TUNING_DS': False,
    }
    global_params = {

    }

    bciciv_1 = load_raw_bciciv_1(ds_params, global_params)

