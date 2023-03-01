import pickle
from tqdm import tqdm
import os
from braindecode.datasets import BaseConcatDataset
import braindecode
import braindecode.datasets.tuh as tuh
from braindecode.preprocessing import create_fixed_length_windows
from torch.utils.data import DataLoader
from mne import set_log_level
import torch
    

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



class SingleChannelWindowsDataset(braindecode.datasets.base.WindowsDataset):
    def __init__(self):
        super(SingleChannelWindowsDataset, self).__init__()
        pass
    # Possible class to use.


def split_channels_and_window():
    # Easy solution, but runs into memory issues
    pass




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

        dataset = SingleChannelDataset(dataset_root, SOURCE_DS)

        with open(cache_path, 'wb') as f:
            pickle.dump(dataset, f)

    # print(ds.description)

    subset = dataset.split(by=range(10))['0']
    print(subset.description)

    subset_windows = create_fixed_length_windows(
        subset,
        picks="eeg",
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=2500,
        window_stride_samples=2500,
        drop_last_window=False,
        # mapping={'M': 0, 'F': 1},  # map non-digit targets
    )
    # store the number of windows required for loading later on
    subset_windows.set_description({
        "n_windows": [len(d) for d in subset_windows.datasets]})  # type: ignore

    # Default DataLoader object lets us iterate through the dataset.
    # Each call to get item returns a batch of samples,
    # each batch has shape: (batch_size, n_channels, n_time_points)
    dl = DataLoader(dataset=subset_windows, batch_size=5)

    batch_X, batch_y, batch_ind = None, None, None
    for batch_X, batch_y, batch_ind in dl:
        pass
    print(batch_X.shape)  # type: ignore
    print('batch_X:', batch_X)
    print('batch_y:', batch_y)
    print('batch_ind:', batch_ind)
