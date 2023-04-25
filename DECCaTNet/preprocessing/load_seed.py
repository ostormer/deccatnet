import glob
import os

from tqdm import tqdm

import mne
from braindecode.datasets import BaseConcatDataset, BaseDataset

def read_raw_seed(dir:str, drop_non_eeg=True):
    file_paths = glob.glob(os.path.join(dir, '**/*.cnt'), recursive=True)
    raws = []
    for file_path in tqdm(file_paths):
        raw = mne.io.read_raw_cnt(file_path, verbose='INFO')
        if drop_non_eeg:
            raw.drop_channels(['M1', 'M2', 'VEO', 'HEO'])
        raws.append(raw)
    for i, raw in enumerate(raws):
        if i < 10:
            print(raw.info)
            print(raw.ch_names)
            print(len(raw.ch_names))
            print(raw.info["dig"])
    return BaseConcatDataset(raws)

if __name__ == '__main__':
    dir = 'D:/SEED/SEED_Multimodal/Chinese/01-EEG-raw'
    concat_ds = read_raw_seed(dir)
    print(concat_ds.datasets)
