import os
import pickle
import yaml

import braindecode.datasets.tuh as tuh
import mne
from braindecode.datautil.serialization import load_concat_dataset

from preprocess import string_to_channel_split_func, window_and_split, select_duration, rename_channels, get_unique_channel_names, first_preprocess_step


if __name__ == "__main__":
    config_path = "preprocessing_oskar.yaml"
    with open(config_path, "r") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    read_cache = params["read_cache"]
    source_ds = params["source_ds"]
    local_load = params["local_load"]
    preproc_params = params["preprocess"]

    window_size = preproc_params["window_size"]
    channel_select_function = preproc_params["channel_select_function"]

    preproc_params = params["preprocess"]
    if (read_cache is None) or read_cache in [False, 'None']:
        read_cache = 'none'
    assert read_cache in ['none', 'raw', 'preproc', 'windows', 'split'], \
        f"{read_cache} is not a valid cache to read"
    assert source_ds in ['tuh_eeg_abnormal', 'tuh_eeg'], \
        f"{source_ds} is not a valid dataset option"
    assert channel_select_function in string_to_channel_split_func.keys(), \
        f"{channel_select_function} is not a valid channel selection function"

    # Disable most MNE logging output which slows execution
    mne.set_log_level(verbose='ERROR')

    if local_load:
        paths = params["directory"]["local"]
    else:
        paths = params["directory"]["disk"]



    if source_ds == 'tuh_eeg_abnormal':
        dataset_root = 'D:/TUH/tuh_eeg_abnormal'
        cache_path = 'D:/TUH/pickles/tuh_abnormal.pkl'
        save_dir = 'D:/TUH/tuh_eeg_abnormal_pre'
        save_dir_2 = 'D:/TUH/tuh_eeg_abnormal_pre2'
        save_dir_indexes = 'D:/TUH/pickles/abnormal_split_indexes.pkl'
        pickle_duration_cache = 'D:/TUH/pickles/tuh_duration.pkl'

    else:
        if local_load:
            dataset_root = r'C:\Users\Styrk\OneDrive - NTNU\Documents\Skole\Master\master_code\master-eeg-trans\datasets\TUH\tuh_eeg\v2.0.0\edf\000'
            cache_path = r'C:\Users\Styrk\OneDrive - NTNU\Documents\Skole\Master\master_code\master-eeg-trans\datasets\TUH_pickles\Styrk-tuh_eeg.pkl'
            save_dir = r'C:\Users\Styrk\OneDrive - NTNU\Documents\Skole\Master\master_code\master-eeg-trans\datasets\TUH\preprocessed\step_1'
            save_dir_2 = r'C:\Users\Styrk\OneDrive - NTNU\Documents\Skole\Master\master_code\master-eeg-trans\datasets\TUH\preprocessed\step_2'
            save_dir_indexes = r'C:\Users\Styrk\OneDrive - NTNU\Documents\Skole\Master\master_code\master-eeg-trans\datasets\TUH\preprocessed/split_indexes.pkl'
            pickle_duration_cache = r'C:\Users\Styrk\OneDrive - NTNU\Documents\Skole\Master\master_code\master-eeg-trans\datasets\TUH_pickles/tuh_duration.pkl'
        else:
            dataset_root = 'D:/TUH/tuh_eeg'
            cache_path = 'D:/TUH/pickles/tuh_eeg.pkl'
            save_dir = 'D:/TUH/tuh_pre'
            save_dir_2 = 'D:/TUH/tuh_pre_2'
            save_dir_indexes = 'D:/TUH/pickles/indexes.pkl'
            pickle_duration_cache = 'D:/TUH/pickles/tuh_eeg_duration.pkl'


    if CACHE_WINDOWS == False:
        if READ_CACHED_RAW_DS:
            with open(cache_path, 'rb') as f:

                dataset = pickle.load(f)
                print('done loading pickled dataset')
        else:
            if source_ds == 'tuh_eeg_abnormal':
                dataset = tuh.TUHAbnormal(dataset_root)
            else:
                dataset = tuh.TUH(dataset_root, n_jobs=8)
            with open(cache_path, 'wb') as f:
                pickle.dump(dataset, f)

        print(f"Loaded {len(dataset.datasets)} files.")
        print(f'Start selecting duration over {MIN_DURATION} sec')
        dataset = select_duration(dataset, t_min=MIN_DURATION, t_max=None)
       # dataset = dataset.split(by=range(50))['0']
        with open(pickle_duration_cache, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Done. Kept {len(dataset.datasets)} files.")
        #dataset = dataset.split(by=range(512))['0']

        print(dataset.description)
        # dataset = get_unique_channel_names(dataset)

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
                              'EEG F3-REF', 'EEG F4-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG FP1-REF', 'EEG FP2-REF', 'EEG FZ-REF',
                              'EEG LUC-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG OZ-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG PZ-REF',
                              'EEG RESP1-REF', 'EEG RESP2-REF', 'EEG RLC-REF', 'EEG SP1-REF', 'EEG SP2-REF', 'EEG T1-REF',
                              'EEG T2-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF'])
        excluded = sorted(
            ["EEG EKG-REF", "EEG ROC-REF", "EEG EKG1-REF", "EEG C3P-REF", "EEG C4P-REF", "EEG LOC-REF", 'EEG EKG-LE',
             'PHOTIC PH', 'DC4-DC', 'DC3-DC', 'DC7-DC', 'DC2-DC', 'DC8-DC', 'DC6-DC', 'DC1-DC', 'DC5-DC', 'EMG-REF',
             'SUPPR', 'IBI', 'PHOTIC-REF', 'BURSTS', 'ECG EKG-REF', 'PULSE RATE', 'RESP ABDOMEN-REF'])

        ar_common_naming = sorted(
            list(set([x.split('-')[0] for x in ar_channels])))
        le_common_naming = sorted(
            list(set([x.split('-')[0] for x in le_channels])))
        common_naming = sorted(list(set(ar_common_naming + le_common_naming)))
        # create dictionaries with key ar or le channel name and send to common name
        ar_to_common = {ar_ref: common for ar_ref,
                        common in zip(ar_channels, ar_common_naming)}
        le_to_common = {le_ref: common for le_ref,
                        common in zip(le_channels, le_common_naming)}
        ch_mapping = {'ar': ar_to_common, 'le': le_to_common}

        # print(common_naming, '\n', le_to_common, '\n', ar_to_common, '\n', ch_mapping)

        tuh_preproc = first_preprocess_step(concat_dataset=dataset, mapping=ch_mapping,
                                            ch_name=common_naming, crop_min=-800, crop_max=800, sfreq=250, save_dir=save_dir, n_jobs=2, )

        ids_to_load = window_and_split(tuh_preproc, save_dir=save_dir_2, overwrite=True,
                                       window_size_samples=15000, n_jobs=2, save_dir_index=save_dir_indexes)
    else:
        with open(save_dir_indexes, 'rb') as f:
            ids_to_load = pickle.load(f)

    windowed_datasets = ContrastiveAugmentedDataset(load_concat_dataset(os.path.join(
        save_dir_2, 'fif_ds'), preload=False, ids_to_load=ids_to_load).datasets) # TODO: think about if target transforms is necessary also add split to get several loaders

    train_split = 0.7
    split_dict = {'test':range(round(len(windowed_datasets.datasets)*(1-train_split))),
                  'train':range(round(len(windowed_datasets.datasets)*(train_split)))}

    splitted = windowed_datasets.split(by=split_dict)
    print(splitted['test'].__len__(), splitted['train'].__len__())
    print(splitted['test'].__len__() + splitted['train'].__len__(),  windowed_datasets.__len__())
    splitted_1 = ContrastiveAugmentedDataset(splitted['0'].datasets)
    print(splitted_1.__len__(), windowed_datasets.__len__())


    print(splitted, windowed_datasets)

    #print(windowed_datasets)
    loader = DataLoader(windowed_datasets, batch_size=10)
    for augmented_1, augmented_2, sample in loader:
        print(augmented_1.shape, augmented_2.shape, sample.shape)

    # how it will look in the end:
    #pre_train_model(windowed_datasets, batch_size=, num_workers=, save_freq=, Shuffel=, model_weights_path=, temperature=,
    #                learning_rate=, weight_decay=, max_epochs=, batch_print_freq=, save_dir_model=, model_file_name=, model_params=)


    # next up: implement some sort of dataloader, general idea for now, create a custom datasetclass, where __getitem__ is overwritten. Then use standard dataloader to iterate through the dtaset

