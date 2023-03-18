
from tqdm import tqdm
import time
from braindecode.datasets import BaseConcatDataset, WindowsDataset
import braindecode.datasets.tuh as tuh
from braindecode.preprocessing import create_fixed_length_windows, Preprocessor, preprocess
from torch.utils.data import DataLoader
from mne import set_log_level
import mne


def make_adjacent_pairs(ch_list: 'list[str]') -> 'list[list[str]]':
    assert len(ch_list) > 1
    pairs = []
    for i in range(len(ch_list)//2):
        pairs.append([ch_list[2*i], ch_list[2*i + 1]])
    if len(ch_list) % 2 == 1:
        pairs.append([ch_list[-1], ch_list[-2]])
    return pairs


def split_channels_and_window(concat_dataset:BaseConcatDataset, channel_split_func=None, window_size_samples=2500) -> BaseConcatDataset:
    """Splits BaseConcatDataset into set containing non-overlapping windows split into channels according to channel_split_func

    Args:
        concat_dataset (braindecode.datasets.BaseConcatDataset): Input dataset
        channel_split_func (callable, optional): Callable function f(ch_names:list[str]) -> list[list[str]]. If None, _make_overlapping_adjacent_pairs is used. Defaults to None.
        window_size_samples (int, optional): Number of time points per window. Defaults to 2500.

    Returns:
        braindecode.datasets.BaseConcatDataset: BaseConcatDataset containing WindowDatasets which have been split up into time windows and channel combinations
    """
    if channel_split_func is None:
        channel_split_func = make_adjacent_pairs
    for base_ds in concat_dataset.datasets:
        base_ds.raw.drop_channels(['IBI', 'BURSTS', 'SUPPR', 'PHOTIC PH'], on_missing='ignore')  # type: ignore
    # Windowing
    t0 = time.time()
    print(f"Begun windowing at {time.ctime(time.time())}")
    windowed_ds = create_fixed_length_windows(
        concat_dataset,
        n_jobs=4,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=window_size_samples,
        window_stride_samples=window_size_samples,
        drop_last_window=True,
    )
    # store the number of windows required for loading later on
    windowed_ds.set_description({
        "n_windows": [len(d) for d in windowed_ds.datasets]})  # type: ignore
    print(f'Finished windowing in {time.time()-t0} seconds')

    all_base_ds = []
    for windows_ds in tqdm(windowed_ds.datasets):
        split_concat_ds = _split_windows_into_channels(windows_ds, channel_split_func)  # type: ignore
        all_base_ds.append(split_concat_ds)

    final_ds = BaseConcatDataset(all_base_ds)
    # print(concat_ds.description)
    return final_ds

def _split_windows_into_channels(base_ds:WindowsDataset, channel_split_func=make_adjacent_pairs) -> BaseConcatDataset:
    raw = base_ds.windows._raw
    channel_selections = channel_split_func(raw.ch_names)

    base_ds_list = []
    old_epochs = base_ds.windows
    raw.load_data()
    for channels in channel_selections:
        new_epochs = mne.Epochs(
            raw=raw,
            events=old_epochs.events,
            picks=channels,
            baseline=None,
            tmin=0,
            tmax=old_epochs.tmax,
            metadata=old_epochs.metadata
        )
        new_epochs.drop_bad()

        ds = WindowsDataset(new_epochs, base_ds.description)
        ds.set_description({"channels": channels})
        base_ds_list.append(ds)

    concat = BaseConcatDataset(base_ds_list)
    return concat



def split_channels_and_window_2(concat_dataset:BaseConcatDataset, channel_split_func=None, window_size_samples=2500) -> BaseConcatDataset:
    """Splits BaseConcatDataset into set containing non-overlapping windows split into channels according to channel_split_func

    Args:
        concat_dataset (braindecode.datasets.BaseConcatDataset): Input dataset
        channel_split_func (callable, optional): Callable function f(ch_names:list[str]) -> list[list[str]]. If None, _make_overlapping_adjacent_pairs is used. Defaults to None.
        window_size_samples (int, optional): Number of time points per window. Defaults to 2500.

    Returns:
        braindecode.datasets.BaseConcatDataset: BaseConcatDataset containing WindowDatasets which have been split up into time windows and channel combinations
    """
    if channel_split_func is None:
        channel_split_func = make_adjacent_pairs
    # Windowing
    t0 = time.time()
    print(f"Begun windowing at {time.ctime(time.time())}")
    windowed_sets = []
    for base_ds in tqdm(concat_dataset.datasets):
        base_ds.raw.drop_channels(['IBI', 'BURSTS', 'SUPPR', 'PHOTIC PH'], on_missing='ignore')  # type: ignore
        picks = channel_split_func(base_ds.raw.ch_names)  # type: ignore
        print(picks)
        for pick in picks:
            single_windowed_ds = create_fixed_length_windows(
                concat_dataset,
                picks=pick,
                start_offset_samples=0,
                stop_offset_samples=None,
                window_size_samples=window_size_samples,
                window_stride_samples=window_size_samples,
                drop_last_window=True,
            )
            if len(single_windowed_ds.datasets[0].windows.ch_names) != len(pick):  # type:ignore
                continue
            # store the number of windows required for loading later on
            single_windowed_ds.set_description({
                "n_windows": [len(d) for d in single_windowed_ds.datasets]})  # type: ignore
            windowed_sets.append(single_windowed_ds)
    print(f'Finished windowing in {time.time()-t0} seconds')

    final_ds = BaseConcatDataset(windowed_sets)
    # print(concat_ds.description)
    return final_ds