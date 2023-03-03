import bisect

from contrastive_framework import pre_train_model
from load_windowed import TUHSingleChannelDataset
import pickle
from tqdm import tqdm
import os
from braindecode.datasets import BaseConcatDataset, WindowsDataset
import braindecode.datasets.tuh as tuh
from braindecode.preprocessing import create_fixed_length_windows
from torch.utils.data import DataLoader
from mne import set_log_level
import torch
from load_windowed import new_getitem, new_len


import collections
import contextlib
import re
import torch

from typing import Callable, Dict, Optional, Tuple, Type, Union
from torch._six import string_classes

np_str_obj_array_pattern = re.compile(r'[SaUO]')


def default_convert(data):
    r"""
        Function that converts each NumPy array element into a :class:`torch.Tensor`. If the input is a `Sequence`,
        `Collection`, or `Mapping`, it tries to convert each element inside to a :class:`torch.Tensor`.
        If the input is not an NumPy array, it is left unchanged.
        This is used as the default function for collation when both `batch_sampler` and
        `batch_size` are NOT defined in :class:`~torch.utils.data.DataLoader`.

        The general input type to output type mapping is similar to that
        of :func:`~torch.utils.data.default_collate`. See the description there for more details.

        Args:
            data: a single data point to be converted

        Examples:
            >>> # Example with `int`
            >>> default_convert(0)
            0
            >>> # Example with NumPy array
            >>> # xdoctest: +SKIP
            >>> default_convert(np.array([0, 1]))
            tensor([0, 1])
            >>> # Example with NamedTuple
            >>> Point = namedtuple('Point', ['x', 'y'])
            >>> default_convert(Point(0, 0))
            Point(x=0, y=0)
            >>> default_convert(Point(np.array(0), np.array(0)))
            Point(x=tensor(0), y=tensor(0))
            >>> # Example with List
            >>> default_convert([np.array([0, 1]), np.array([2, 3])])
            [tensor([0, 1]), tensor([2, 3])]
    """
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        # array of string classes and object
        if elem_type.__name__ == 'ndarray' \
                and np_str_obj_array_pattern.search(data.dtype.str) is not None:
            return data
        return torch.as_tensor(data)
    elif isinstance(data, collections.abc.Mapping):
        try:
            return elem_type({key: default_convert(data[key]) for key in data})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: default_convert(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(default_convert(d) for d in data))
    elif isinstance(data, tuple):
        return [default_convert(d) for d in data]  # Backwards compatibility.
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, string_classes):
        try:
            return elem_type([default_convert(d) for d in data])
        except TypeError:
            # The sequence type may not support `__init__(iterable)` (e.g., `range`).
            return [default_convert(d) for d in data]
    else:
        return data


default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def collate(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    r"""
        General collate function that handles collection type of element within each batch
        and opens function registry to deal with specific element types. `default_collate_fn_map`
        provides default collate functions for tensors, numpy arrays, numbers and strings.

        Args:
            batch: a single batch to be collated
            collate_fn_map: Optional dictionary mapping from element type to the corresponding collate function.
              If the element type isn't present in this dictionary,
              this function will go through each key of the dictionary in the insertion order to
              invoke the corresponding collate function if the element type is a subclass of the key.

        Examples:
            >>> # Extend this function to handle batch of tensors
            >>> def collate_tensor_fn(batch, *, collate_fn_map):
            ...     return torch.stack(batch, 0)
            >>> def custom_collate(batch):
            ...     collate_map = {torch.Tensor: collate_tensor_fn}
            ...     return collate(batch, collate_fn_map=collate_map)
            >>> # Extend `default_collate` by in-place modifying `default_collate_fn_map`
            >>> default_collate_fn_map.update({torch.Tensor: collate_tensor_fn})

        Note:
            Each collate function requires a positional argument for batch and a keyword argument
            for the dictionary of collate functions as `collate_fn_map`.
    """
    elem = batch[0]
    elem_type = type(elem)

    if collate_fn_map is not None:
        if elem_type in collate_fn_map:
            return collate_fn_map[elem_type](batch, collate_fn_map=collate_fn_map)

        for collate_type in collate_fn_map:
            if isinstance(elem, collate_type):
                return collate_fn_map[collate_type](batch, collate_fn_map=collate_fn_map)

    if isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: collate([d[key] for d in batch], collate_fn_map=collate_fn_map) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: collate([d[key] for d in batch], collate_fn_map=collate_fn_map) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate(samples, collate_fn_map=collate_fn_map) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [collate(samples, collate_fn_map=collate_fn_map) for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type([collate(samples, collate_fn_map=collate_fn_map) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [collate(samples, collate_fn_map=collate_fn_map) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def collate_tensor_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    elem = batch[0]
    out = None
    if torch.utils.data.get_worker_info() is not None:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum(x.numel() for x in batch)
        storage = elem.storage()._new_shared(numel, device=elem.device)
        out = elem.new(storage).resize_(len(batch), *list(elem.size()))
    return torch.stack(batch, 0, out=out)


def collate_numpy_array_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    elem = batch[0]
    # array of string classes and object
    if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
        raise TypeError(default_collate_err_msg_format.format(elem.dtype))

    return collate([torch.as_tensor(b) for b in batch], collate_fn_map=collate_fn_map)


def collate_numpy_scalar_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    return torch.as_tensor(batch)


def collate_float_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    return torch.tensor(batch, dtype=torch.float64)


def collate_int_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    return torch.tensor(batch)


def collate_str_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    return batch


default_collate_fn_map: Dict[Union[Type, Tuple[Type, ...]], Callable] = {torch.Tensor: collate_tensor_fn}
with contextlib.suppress(ImportError):
    import numpy as np
    # For both ndarray and memmap (subclass of ndarray)
    default_collate_fn_map[np.ndarray] = collate_numpy_array_fn
    # See scalars hierarchy: https://numpy.org/doc/stable/reference/arrays.scalars.html
    # Skip string scalars
    default_collate_fn_map[(np.bool_, np.number, np.object_)] = collate_numpy_scalar_fn
default_collate_fn_map[float] = collate_float_fn
default_collate_fn_map[int] = collate_int_fn
default_collate_fn_map[string_classes] = collate_str_fn


def default_collate(batch):
    r"""
        Function that takes in a batch of data and puts the elements within the batch
        into a tensor with an additional outer dimension - batch size. The exact output type can be
        a :class:`torch.Tensor`, a `Sequence` of :class:`torch.Tensor`, a
        Collection of :class:`torch.Tensor`, or left unchanged, depending on the input type.
        This is used as the default function for collation when
        `batch_size` or `batch_sampler` is defined in :class:`~torch.utils.data.DataLoader`.

        Here is the general input type (based on the type of the element within the batch) to output type mapping:

            * :class:`torch.Tensor` -> :class:`torch.Tensor` (with an added outer dimension batch size)
            * NumPy Arrays -> :class:`torch.Tensor`
            * `float` -> :class:`torch.Tensor`
            * `int` -> :class:`torch.Tensor`
            * `str` -> `str` (unchanged)
            * `bytes` -> `bytes` (unchanged)
            * `Mapping[K, V_i]` -> `Mapping[K, default_collate([V_1, V_2, ...])]`
            * `NamedTuple[V1_i, V2_i, ...]` -> `NamedTuple[default_collate([V1_1, V1_2, ...]),
              default_collate([V2_1, V2_2, ...]), ...]`
            * `Sequence[V1_i, V2_i, ...]` -> `Sequence[default_collate([V1_1, V1_2, ...]),
              default_collate([V2_1, V2_2, ...]), ...]`

        Args:
            batch: a single batch to be collated

        Examples:
            >>> # Example with a batch of `int`s:
            >>> default_collate([0, 1, 2, 3])
            tensor([0, 1, 2, 3])
            >>> # Example with a batch of `str`s:
            >>> default_collate(['a', 'b', 'c'])
            ['a', 'b', 'c']
            >>> # Example with `Map` inside the batch:
            >>> default_collate([{'A': 0, 'B': 1}, {'A': 100, 'B': 100}])
            {'A': tensor([  0, 100]), 'B': tensor([  1, 100])}
            >>> # Example with `NamedTuple` inside the batch:
            >>> # xdoctest: +SKIP
            >>> Point = namedtuple('Point', ['x', 'y'])
            >>> default_collate([Point(0, 0), Point(1, 1)])
            Point(x=tensor([0, 1]), y=tensor([0, 1]))
            >>> # Example with `Tuple` inside the batch:
            >>> default_collate([(0, 1), (2, 3)])
            [tensor([0, 2]), tensor([1, 3])]
            >>> # Example with `List` inside the batch:
            >>> default_collate([[0, 1], [2, 3]])
            [tensor([0, 2]), tensor([1, 3])]
            >>> # Two options to extend `default_collate` to handle specific type
            >>> # Option 1: Write custom collate function and invoke `default_collate`
            >>> def custom_collate(batch):
            ...     elem = batch[0]
            ...     if isinstance(elem, CustomType):  # Some custom condition
            ...         return ...
            ...     else:  # Fall back to `default_collate`
            ...         return default_collate(batch)
            >>> # Option 2: In-place modify `default_collate_fn_map`
            >>> def collate_customtype_fn(batch, *, collate_fn_map=None):
            ...     return ...
            >>> default_collate_fn_map.update(CustoType, collate_customtype_fn)
            >>> default_collate(batch)  # Handle `CustomType` automatically
    """
    return collate(batch, collate_fn_map=default_collate_fn_map)


if __name__=="__main__":
    # the goal for this iteration is to run pre_train_model function
    # still need a way to load the dataset,
    READ_CACHED_DS = False # Change to read cache or not
    SOURCE_DS = 'tuh_eeg'  # Which dataset to load

    assert SOURCE_DS in ['tuh_eeg_abnormal', 'tuh_eeg']
    # Disable most MNE logging output which slows execution
    set_log_level(verbose='WARNING')

    dataset_root = None
    cache_path = None
    dataset = None
    if SOURCE_DS == 'tuh_eeg_abnormal':
        dataset_root = '../datasets/TUH/tuh_eeg_abnormal'
        # remeber to choose correct cache part for your computer, or drive, in addition tuh.py does not work for mye
        # for tuh_eeg_abnormal
        cache_path = '../datasets/tuh_braindecode/styrk_tuh_abnormal.pkl'

    else:
        dataset_root = '../datasets/TUH/tuh_eeg'
        cache_path = '../datasets/tuh_braindecode/styrk_tuh_eeg.pkl'

    # since the pickel files are references to locations on a disk, traversing between computers is hard. However
    # It is possible to do this with

    if READ_CACHED_DS:
        with open(cache_path, 'rb') as f:
            dataset = pickle.load(f)
    else:
        print('creating TUHSingleChannelDataset')
        dataset = TUHSingleChannelDataset(path=dataset_root, source_dataset=SOURCE_DS)
        with open(cache_path, 'wb') as f:
            pickle.dump(dataset, f)


    w_s_samples = 2000
    windowed_ds = create_fixed_length_windows(
        dataset,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=w_s_samples,
        window_stride_samples=2000,
        drop_last_window=True,
    )
    # store the number of windows required for loading later on
    windowed_ds.set_description({
        "n_windows": [len(d) for d in windowed_ds.datasets]})
    #print(ds.description)
    #subset = dataset.split(by=range(10))['0']
    """
    NB!! important to not create a subset as this creates a new class which has a new
    __getitem__ function which overrides the original. 
    """
    windowed_ds.__getitem__ = new_getitem
    windowed_ds.__len__ = new_len
    print(windowed_ds.__getitem__(windowed_ds,1).shape)
    dl = DataLoader(dataset=windowed_ds, batch_size=10)

    batch_X, batch_y, batch_ind = None, None, None
    for something in dl:
        """
        The iteration stops when method __getitem__ raises IndexError for some index idx
        """
        print(something.shape)
        break
    print(batch_X.shape)  # type: ignore
   # print('batch_X:', batch_X)
    print('batch_y:', batch_y)
    print('batch_ind:', batch_ind)

    #pre_train_model(batch_size=100, num_workers=1,save_freq=10,Shuffel=False,save_dir_model='models',model_file_name='test',model_weights_dict=None,temperature= 2
    #               ,learning_rate= 0.01
    #               , weight_decay= 0.01,max_epochs=20,batch_print_condition=5)

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


def new_len(self: BaseConcatDataset):
    # TODO: need to find a way to find correct len
    counter = []
    for dataset in self.datasets:
        counter.append(len(dataset.windows.ch_names))
    cum_sizes = self.cumsum(self.datasets)
    for i in len(cum_sizes):
        if i == 0:
            cum_sizes[0] = counter[0] * cum_sizes[0]
        else:
            cum_sizes[i] = cum_sizes[i - 1] + cum_sizes[i] * counter[i]
    self.cumulative_sizes = cum_sizes
    return cum_sizes[-1] * counter[-1]


def new_getitem(self: BaseConcatDataset, idx):
    """
    data is stored in self.datasets, so self.datasets is a concated dataset of baseDatasets types.
    BaseDataset is also a braindecode extension of pytroch Dataset.
    Could mean that we should not overwrite this __getitem__, but the one for the base datasets.
    :param index:
    :return:
    """
    if idx < 0:
        if -idx > len(self):
            raise ValueError("absolute value of index should not exceed dataset length")
        idx = len(self) + idx
    dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)

    self.n_channels = len(self.datasets[dataset_idx].windows.ch_names)

    if dataset_idx == 0:
        sample_idx = idx
    else:
        sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
    new_sample_idx = sample_idx // self.n_channels
    # want to find recording_idx from idx
    if new_sample_idx == 0:
        channel_idx = sample_idx
    else:
        channel_idx = sample_idx - new_sample_idx * self.n_channels

    # print(self.datasets[dataset_idx][new_sample_idx][0].shape)
    print('using the monkey patched __getitem__ function')

    return self.datasets[dataset_idx][new_sample_idx][0][channel_idx].reshape(1, -1)


class TUHSingleChannelDataset(tuh.TUH):
    def __init__(self, path, source_dataset, recording_ids=None, target_name=None,
                 preload=False, add_physician_reports=False, n_jobs=1):

        super().__init__(path=path, recording_ids=recording_ids,
                         preload=preload, target_name=None,
                         add_physician_reports=add_physician_reports,
                         n_jobs=n_jobs)
        self.cumulative_sizes = self.cumsum(self.datasets)
        self.n_channels = self.datasets[0][0][0].shape[0]
        # we sample each chanel with the next channel, and the last one with the first
        # so now we go channel-wise
        # print(self.datasets[0][0].get_sequence)
        # self.cumulative_sizes = [i * self.n_channels for i in self.cumulative_sizes]
        assert len(self.datasets) == len(self.cumulative_sizes)

    def __len__(self):
        """
        important to overwrtie __len__ aswell as the
        :return:
        """
        return self.cumulative_sizes[-1]

    """
    This is how __len__() is inherited from the base file
    self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]
    """

    def __getitem__(self, idx):
        """
    data is stored in self.datasets, so self.datasets is a concated dataset of baseDatasets types.
    BaseDataset is also a braindecode extension of pytroch Dataset.
    Could mean that we should not overwrite this __getitem__, but the one for the base datasets.
        :param index:
        :return:
    """
        # access one and one dataset and rewrite the __getitem__ function of the
        # downstream dataset to return two and two or one and one sample?
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        # get dataset
        # dataset = self.datasets[dataset_idx]
        # create cumulative sizes for sample
        # print(dataset.)
        # cumulative_sample_sizes = dataset
        # print(cumulative_sample_sizes)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        new_sample_idx = sample_idx // self.n_channels
        # want to find recording_idx from idx
        if new_sample_idx == 0:
            recording_idx = sample_idx
        else:
            recording_idx = sample_idx - new_sample_idx * self.n_channels

        # now we have a way to iterate through each dataset
        # need a way to exstract two and two channels
        # one_recording = self.datasets[dataset_idx][0][0] #this is one recording
        # now self.cumulative sizes is n_channels as large. We need to get sample_idx
        # correct, hopefully
        # print(one_recording.shape, self.n_channels)
        # self.cumsum(one_recording)
        # print(self.datasets[dataset_idx][new_sample_idx])
        """
        This appoarch doesnt work now as __getitem__ returns
        one and one datapoint when not windowed
        """
        print(self.datasets[dataset_idx].shape, self.datasets[dataset_idx][new_sample_idx].shape)
        print(self.datasets[dataset_idx][new_sample_idx][recording_idx].shape)
        return self.datasets[dataset_idx][new_sample_idx][0][recording_idx]

    """
    Thoughts: we have to read the data twice, one time when we pre-process and when we train
        the model. Applying augmentations during pre-processing doesnt seam optimal.

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

    """ getitem of BaseConcat, so super for a TUH dataset
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