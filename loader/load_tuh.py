import os
import numpy as np
from pyedflib import EdfReader
from tqdm import tqdm

from sample import Sample

USED_CHANNELS = [
    "EEG FP1-REF",
    "EEG FP2-REF",
    "EEG F3-REF",
    "EEG F4-REF",
    "EEG C3-REF",
    "EEG C4-REF",
    "EEG P3-REF",
    "EEG P4-REF",
    "EEG O1-REF",
    "EEG O2-REF",
    "EEG F7-REF",
    "EEG F8-REF",
    "EEG T3-REF",
    "EEG T4-REF",
    "EEG T5-REF",
    "EEG T6-REF",
    "EEG A1-REF",
    "EEG A2-REF",
    "EEG FZ-REF",
    "EEG CZ-REF",
    "EEG PZ-REF",
    "EEG EKG1-REF",
    "EEG T1-REF",
    "EEG T2-REF",
    # "IBI",
    # "BURSTS",
    # "SUPPR",
]




def read_single_edf(file_path):
    file = EdfReader(file_path)
    channel_names = file.getSignalLabels()
    print(file.getNSamples(), file.getSampleFrequencies())
    signal_buffer = np.zeros((len(USED_CHANNELS), file.getNSamples()))
    buffer_i = 0
    for i, channel in enumerate(channel_names):
        if channel in USED_CHANNELS:
            signal_buffer[buffer_i, :] = file.readSignal(i)
            buffer_i +=  1

    return channel_names


def walk_sub_dirs(path):
    channel_counts = {}
    for _dir_path, _dir_names, file_names in tqdm(os.walk(path)):
        for f in file_names:
            if f.endswith(".edf"):
                pass
    return channel_counts


def read_edf_channel_names(file_path):
    file = EdfReader(file_path)
    return file.getSignalLabels()


def count_sub_dirs_channels(path):
    channel_counts = {}
    for dir_path, _dir_names, file_names in tqdm(os.walk(path)):
        for f in file_names:
            if f.endswith(".edf"):
                channels = read_single_edf(os.path.join(dir_path, f))
                for c in channels:
                    if c in channel_counts.keys():
                        channel_counts[c] += 1
                    else:
                        channel_counts[c] = 1
    return channel_counts


if __name__ == "__main__":

    FP = "tuh-test"
    channel_counts = count_sub_dirs_channels(FP)
    for channel, count in channel_counts.items():
        print(f"{channel} : {count}")