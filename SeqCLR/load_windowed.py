import pickle
from tqdm import tqdm
import time
from braindecode.datasets import BaseConcatDataset, BaseDataset
import braindecode.datasets.tuh as tuh
from braindecode.preprocessing import create_fixed_length_windows
from torch.utils.data import DataLoader
from mne import set_log_level
import mne
import braindecode
import torch
import torch.utils.data
    

'''
# I don't think SingleChannelDataset is necessary, it would be better to split into single channels
# or channel pairs while/after/before windowing.
# If we want a custom dataset object to handle augmentations etc, we can make a wrapper
# class that contains a Dataset or BaseConcatDataset or whatever and can get items and apply augmentations to it
class SingleChannelDataset(tuh.TUHAbnormal):
    def __init__(self, path, source_dataset, recording_ids=None, target_name=None,
                 preload=False, add_physician_reports=False, n_jobs=1):
        if source_dataset == "tuh_eeg_abnormal":
            print("Initializing TUHAbnormal object")
            tuh.TUHAbnormal.__init__(self, path=path, recording_ids=recording_ids,
                                     preload=preload, target_name='pathological',
                                     add_physician_reports=add_physician_reports,
                                     n_jobs=n_jobs)
        elif source_dataset == "tuh_eeg":
            print("Initializing TUH object")
            tuh.TUH.__init__(self, path=path, recording_ids=recording_ids,
                             preload=preload, target_name=None,
                             add_physician_reports=add_physician_reports,
                             n_jobs=n_jobs)
        else:
            print(f"Dataset type <{source_dataset}> has not been implemented")
            raise NotImplementedError

    def __getitem__(self, idx):
        return super().__getitem__(idx)
        # Don't know how this one would read single channels from each file on request.
        # Will take a look on extending WindowedDataset instead 

        # Iterating without shuffling could be supported, only one file would 
        # have to be read at a time and kept open in memory. However if indexing
        # and shuffling should be supported, a list of number of channels would
        # need to be stored together with the paths, and indexes would have to be
        # the total cumulative channel number
        # 
        # This is possible, though each chanel read would require to load and unload
        # a edf file when shuffling.
        # Would be just as efficient to just extract a single channel from the 
        # WindowsDataset 
'''


class TUHSingleChannelDataset(tuh.TUH):
    def __init__(self, path, source_dataset, recording_ids=None, target_name=None,
                 preload=False, add_physician_reports=False, n_jobs=1):

        super().__init__(path=path, recording_ids=recording_ids,
                             preload=preload, target_name=None,
                             add_physician_reports=add_physician_reports,
                             n_jobs=n_jobs)
        print(len(self.datasets)) # 35 datasets right now
        print(type(self))

    def __getitem__(self, index):
        """
    data is stored in self.datasets, so self.datasets is a concated dataset of baseDatasets types.
    BaseDataset is also a braindecode extension of pytroch Dataset.
    Could mean that we should not overwrite this __getitem__, but the one for the base datasets.
        :param index:
        :return:
    """
        return index# first test with super

    """
    Hirearchy:
    TUHDataset is an extension of BaseConcatDataset which a concatination of several BaseDataset. 
    However BaseConcatDataset is an extension of Pytorch ConcatDataset. 
    BaseDataset is an extension of Pytorch Dataset, however few or no super() calls is utilized.
    """

    """    getitem of base
    def __getitem__(self, index):
        X = self.raw[:, index][0]
        y = None
        if self.target_name is not None:
            y = self.description[self.target_name]
        if isinstance(y, pd.Series):
            y = y.to_list()
        if self.transform is not None:
            X = self.transform(X)
        return X, y
        
    
    __getitem__ of basic dataset
    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)
    """

    """ getitem of concat, so super for a TUH dataset
    def __getitem__(self, idx):
        Parameters
        ----------
        idx : int | list
            Index of window and target to return. If provided as a list of
            ints, multiple windows and targets will be extracted and
            concatenated. The target output can be modified on the
            fly by the ``traget_transform`` parameter.
        if isinstance(idx, Iterable):  # Sample multiple windows
            item = self._get_sequence(idx)
        else:
            item = super().__getitem__(idx)
        if self.target_transform is not None:
            item = item[:1] + (self.target_transform(item[1]),) + item[2:]
        return item
        
    def _get_sequence(self, indices):
        X, y = list(), list()
        for ind in indices:
            out_i = super().__getitem__(ind)
            X.append(out_i[0])
            y.append(out_i[1])

        X = np.stack(X, axis=0)
        y = np.array(y)

        return X, y
    
    getitem of pytroch concat dataset, used in the two above
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]
    
    """



def split_channels_and_window(dataset:BaseConcatDataset, mode='single', window_size_samples=2500) -> BaseConcatDataset:

    assert mode in ['single', 'pairs'], f'{mode} not a valid spliting strategy'

    all_base_ds = []
    for base_ds in tqdm(dataset.datasets):
        base_ds.raw.drop_channels(['IBI', 'BURSTS', 'SUPPR'], on_missing='ignore')
        if base_ds.raw.n_times < window_size_samples:
            # Drop short recordings
            continue
        if mode == 'single':
            split_concat_ds = _split_into_single_channels(base_ds)
        else: #  mode == 'pairs':
            split_concat_ds = _split_into_all_channel_pairs(base_ds, unique_pairs=True)
    
        # Create list of BaseDatasets containing one split_raw sample each
        all_base_ds.append(split_concat_ds)
        
    concat_ds = BaseConcatDataset(all_base_ds)
    # print(concat_ds.description)
    t0 = time.time()
    print(f"Begun windowing at {time.ctime(time.time())}")
    windowed_ds = create_fixed_length_windows(
        concat_ds,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=window_size_samples,
        window_stride_samples=2500,
        drop_last_window=False,
    )
    # store the number of windows required for loading later on
    windowed_ds.set_description({
        "n_windows": [len(d) for d in windowed_ds.datasets]})  # type: ignore
    print(f'Finished windowing in {time.time()-t0} seconds')
    return windowed_ds

def _split_into_single_channels(base_ds:BaseDataset) -> BaseConcatDataset:
    base_ds_list = []
    raw = base_ds.raw
    for channel in raw.ch_names:
        new_raw = raw.copy().pick_channels([channel])
        ds = BaseDataset(new_raw, description=base_ds.description)
        ds.set_description({"Channels": channel})

        base_ds_list.append(ds)
    concat = BaseConcatDataset(base_ds_list)
    return concat


def _split_into_all_channel_pairs(base_ds:BaseDataset, unique_pairs=True) -> BaseConcatDataset:
    pairs = []
    raw = base_ds.raw
    if not unique_pairs:
        for channel_i in raw.ch_names[:]:
            for channel_j in raw.ch_names[:]:
                pairs.append([channel_i, channel_j])
    else:  # Only unique pairs, i.e. not (i,j) if (j,i) exists, and not (i,i)
        for i, channel_i in enumerate(raw.ch_names[:]):
            for j, channel_j in enumerate(raw.ch_names[i+1:]):
                pairs.append([channel_i, channel_j])
    
    base_ds_list = []
    for pair in pairs:
        new_raw = raw.copy().pick_channels(pair)
        ds = BaseDataset(new_raw, description=base_ds.description)
        ds.set_description({"Channels": pair})
        base_ds_list.append(ds)
        
    concat = BaseConcatDataset(base_ds_list)
    return concat




if __name__ == "__main__":
    READ_CACHED_DS = False  # Change to read cache or not
    SOURCE_DS = 'tuh_eeg'  # Which dataset to load

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
        dataset_root = 'datasets/tuh_test/tuh_eeg'
        cache_path = 'datasets/tuh_braindecode/tuh_eeg.pkl'

    if READ_CACHED_DS:
        with open(cache_path, 'rb') as f:

            dataset = pickle.load(f)
    else:

        if SOURCE_DS == 'tuh_eeg_abnormal':
            dataset = tuh.TUHAbnormal(dataset_root)

        else:
            dataset = tuh.TUH(dataset_root)

        with open(cache_path, 'wb') as f:
            pickle.dump(dataset, f)

    print(dataset.description)

    print('Loaded DS')

    single_channel_windows = split_channels_and_window(dataset, mode='single')

    print("Windowing complete")
    # Default DataLoader object lets us iterate through the dataset.
    # Each call to get item returns a batch of samples,
    # each batch has shape: (batch_size, n_channels, n_time_points)
    loader = DataLoader(dataset=single_channel_windows, batch_size=32)

    batch_X, batch_y, batch_ind = None, None, None
    for batch_X, batch_y, batch_ind in loader:
        pass
    print(batch_X.shape)  # type: ignore
    print('batch_X:', batch_X)
    print('batch_y:', batch_y)
    print('batch_ind:', batch_ind)
