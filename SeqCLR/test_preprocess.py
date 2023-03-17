import bisect
import pickle
from tqdm import tqdm
import time
from braindecode.datasets import BaseConcatDataset, WindowsDataset
import braindecode.datasets.tuh as tuh
from braindecode.preprocessing import create_fixed_length_windows
from torch.utils.data import DataLoader
from mne import set_log_level
import mne
from load_windowed import select_duration, rename_channels, get_unique_channel_names, first_preprocess_step

if __name__ == "__main__":
    READ_CACHED_DS = True  # Change to read cache or not
    SOURCE_DS = 'tuh_eeg'  # Which dataset to load
    LOCAL_LOAD = False

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
        if LOCAL_LOAD:
            dataset_root = '../datasets/TUH/tuh_eeg'
            cache_path = '../datasets/tuh_braindecode/tuh_Styrk.pkl'
        else:
            dataset_root = 'D:/TUH/tuh_eeg'
            cache_path = 'D:/TUH/pickles/tuh_eeg'

    if READ_CACHED_DS:
        with open(cache_path, 'rb') as f:

            dataset = pickle.load(f)
            print('done loading pickled dataset')
    else:

        if SOURCE_DS == 'tuh_eeg_abnormal':
            dataset = tuh.TUHAbnormal(dataset_root)

        else:
            dataset = tuh.TUH(dataset_root, n_jobs=2)
            print('done creating TUH dataset')
        with open(cache_path, 'wb') as f:
            pickle.dump(dataset, f)
        print('done pickling')

    dataset = select_duration(dataset, t_min=10, t_max=1000)
    with open('D:/TUH/pickles/tuh_eeg_duration', 'wb') as f:
        pickle.dump(dataset, f)
    print('done selecting duration')
    #dataset = get_unique_channel_names(dataset)

    # create mapping from channel names to channel
    le_channels = sorted(['EEG 20-LE', 'EEG 21-LE', 'EEG 22-LE', 'EEG 23-LE', 'EEG 24-LE', 'EEG 25-LE', 'EEG 26-LE',
                          'EEG 27-LE', 'EEG 28-LE', 'EEG 29-LE', 'EEG 30-LE', 'EEG 31-LE', 'EEG 32-LE', 'EEG A1-LE',
                          'EEG A2-LE', 'EEG C3-LE', 'EEG C4-LE', 'EEG CZ-LE', 'EEG F3-LE', 'EEG F4-LE', 'EEG F7-LE',
                          'EEG F8-LE',
                          'EEG FP1-LE', 'EEG FP2-LE', 'EEG FZ-LE', 'EEG LUC-LE', 'EEG O1-LE', 'EEG O2-LE', 'EEG OZ-LE',
                          'EEG P3-LE', 'EEG P4-LE', 'EEG PG1-LE', 'EEG PG2-LE', 'EEG PZ-LE', 'EEG RLC-LE', 'EEG SP1-LE',
                          'EEG SP2-LE', 'EEG T1-LE', 'EEG T2-LE', 'EEG T3-LE', 'EEG T4-LE', 'EEG T5-LE',
                          'EEG T6-LE'])
    ar_channels = sorted(['EEG 100-REF', 'EEG 101-REF', 'EEG 102-REF', 'EEG 103-REF', 'EEG 104-REF', 'EEG 105-REF',
                          'EEG 106-REF', 'EEG 107-REF', 'EEG 108-REF', 'EEG 109-REF', 'EEG 110-REF', 'EEG 111-REF',
                          'EEG 112-REF', 'EEG 113-REF', 'EEG 114-REF', 'EEG 115-REF', 'EEG 116-REF', 'EEG 117-REF',
                          'EEG 118-REF', 'EEG 119-REF', 'EEG 120-REF', 'EEG 121-REF', 'EEG 122-REF', 'EEG 123-REF',
                          'EEG 124-REF', 'EEG 125-REF', 'EEG 126-REF', 'EEG 127-REF', 'EEG 128-REF', 'EEG 20-REF',
                          'EEG 21-REF', 'EEG 22-REF', 'EEG 23-REF', 'EEG 24-REF', 'EEG 25-REF', 'EEG 26-REF',
                          'EEG 27-REF',
                          'EEG 28-REF', 'EEG 29-REF', 'EEG 30-REF', 'EEG 31-REF', 'EEG 32-REF', 'EEG 33-REF',
                          'EEG 34-REF',
                          'EEG 35-REF', 'EEG 36-REF', 'EEG 37-REF', 'EEG 38-REF', 'EEG 39-REF', 'EEG 40-REF',
                          'EEG 41-REF',
                          'EEG 42-REF', 'EEG 43-REF', 'EEG 44-REF', 'EEG 45-REF', 'EEG 46-REF', 'EEG 47-REF',
                          'EEG 48-REF',
                          'EEG 49-REF', 'EEG 50-REF', 'EEG 51-REF', 'EEG 52-REF', 'EEG 53-REF', 'EEG 54-REF',
                          'EEG 55-REF',
                          'EEG 56-REF', 'EEG 57-REF', 'EEG 58-REF', 'EEG 59-REF', 'EEG 60-REF', 'EEG 61-REF',
                          'EEG 62-REF',
                          'EEG 63-REF', 'EEG 64-REF', 'EEG 65-REF', 'EEG 66-REF', 'EEG 67-REF', 'EEG 68-REF',
                          'EEG 69-REF',
                          'EEG 70-REF', 'EEG 71-REF', 'EEG 72-REF', 'EEG 73-REF', 'EEG 74-REF', 'EEG 75-REF',
                          'EEG 76-REF',
                          'EEG 77-REF', 'EEG 78-REF', 'EEG 79-REF', 'EEG 80-REF', 'EEG 81-REF', 'EEG 82-REF',
                          'EEG 83-REF',
                          'EEG 84-REF', 'EEG 85-REF', 'EEG 86-REF', 'EEG 87-REF', 'EEG 88-REF', 'EEG 89-REF',
                          'EEG 90-REF',
                          'EEG 91-REF', 'EEG 92-REF', 'EEG 93-REF', 'EEG 94-REF', 'EEG 95-REF', 'EEG 96-REF',
                          'EEG 97-REF',
                          'EEG 98-REF', 'EEG 99-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG C3-REF', 'EEG C4-REF',
                          'EEG CZ-REF',
                          'EEG F3-REF', 'EEG F4-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG FP1-REF', 'EEG FP2-REF',
                          'EEG FZ-REF',
                          'EEG LUC-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG OZ-REF', 'EEG P3-REF', 'EEG P4-REF',
                          'EEG PZ-REF',
                          'EEG RESP1-REF', 'EEG RESP2-REF', 'EEG RLC-REF', 'EEG SP1-REF', 'EEG SP2-REF', 'EEG T1-REF',
                          'EEG T2-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF'])
    excluded = sorted(
        ["EEG EKG-REF", "EEG ROC-REF", "EEG EKG1-REF", "EEG C3P-REF", "EEG C4P-REF", "EEG LOC-REF", 'EEG EKG-LE',
         'PHOTIC PH', 'DC4-DC', 'DC3-DC', 'DC7-DC', 'DC2-DC', 'DC8-DC', 'DC6-DC', 'DC1-DC', 'DC5-DC', 'EMG-REF',
         'SUPPR', 'IBI', 'PHOTIC-REF', 'BURSTS', 'ECG EKG-REF', 'PULSE RATE', 'RESP ABDOMEN-REF'])

    ar_common_naming = sorted(list(set([x.split('-')[0] for x in ar_channels])))
    le_common_naming = sorted(list(set([x.split('-')[0] for x in le_channels])))
    common_naming = sorted(list(set(ar_common_naming + le_common_naming)))
    # create dictionaries with key ar or le channel name and send to common name
    ar_to_common = {ar_ref:common for ar_ref,common in zip(ar_channels, ar_common_naming)}
    le_to_common = {le_ref:common for le_ref,common in zip(le_channels, le_common_naming)}
    ch_mapping = {'ar': ar_to_common, 'le':le_to_common}

    # print(common_naming, '\n', le_to_common, '\n', ar_to_common, '\n', ch_mapping)
    save_dir = 'D:/TUH/tuh_pre'
    tuh_preproc = first_preprocess_step(concat_dataset=dataset, mapping=ch_mapping,ch_name=common_naming,crop_min=0, crop_max=1, sfreq=250,save_dir=save_dir,n_jobs=2, )
