from random import randint, random
import numpy as np
from scipy import signal

import matplotlib.pyplot as plt

from mne.io.edf import read_raw_edf


def amplitude_scale(x, max_scale):
    """Rescale. Equal chance of upscale or downscale,
    scaling factor chosen uniformly, between 1 and max_scale

    Args:
        x (array): 1d signal to rescale
        max_scale (float): factor by which to scale signal up or down

    Returns:
        array: scaled 1d signal
    """
    factor = 1 + random()*(max_scale-1)
    if randint(0, 1):
        return x * factor  # Upscale
    else:
        return x / factor  # Downscale


def time_shift(x, min_shift, max_shift):
    """Time shift the signal. Signal contents get rolled start-to-end and oposite

    Args:
        x (array): 1d signal
        min_shift (int): min samples to shift by
        max_shift (int): max samples to shift by

    Returns:
        array: shifted 1d signal
    """
    shift = randint(min_shift, max_shift)
    return np.roll(x, shift)


def dc_shift(x, min_shift, max_shift):
    """Shift signal by adding constant value

    Args:
        x (array): 1d signal
        min_shift (float): min shift
        max_shift (float): max shift

    Returns:
        array: shifted 1d signal
    """
    return x + min_shift + random()*(max_shift-min_shift)


def zero_mask(x, min_size, max_size, min_pad):
    """Mask portion of signal by setting it to 0

    Args:
        x (array): 1d signal
        min_size (int): min n samples to mask
        max_size (int): max n samples to mask, overridden by min_pad if it does not fit
        min_pad (int, optional): minimum of space to leave at each end of the mask. Defaults to 0.

    Returns:
        array: masked 1d signal
    """
    # Make sure max size fits within the signal length, including paddings
    max_size = min(max_size, len(x) + 2*min_pad)
    min_size = min(min_size, max_size)
    size = randint(min_size, max_size)
    mask_start = randint(min_pad, len(x) - size - min_pad)
    x[mask_start:mask_start+size] = np.zeros(size)
    # TODO: Test edge cases and that length is preserved
    return x


def add_gaussian_noise(x, min_sd, max_sd):
    """Augmentation by additive gaussian noise, SD chosen randomly

    Args:
        x (array): 1d signal
        min_sd (float): min standard deviation
        max_sd (float): max standard deviation

    Returns:
        array: augmented 1d signal
    """
    sd = min_sd + random()*(max_sd-min_sd)
    noise = np.random.normal(0, sd, x.shape)
    return x + noise


def band_stop_filter(x, sig_freq, min_freq, max_freq):
    # MNE does this: https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.notch_filter
    # TODO: Implement this using scipy signal
    # see https://stackoverflow.com/questions/50247517/python-specific-frequency-removenotch-filter/63038706#63038706
    stop_freq = min_freq + random()*(max_freq-min_freq)
    notch_width = 5
    quality_factor = notch_width / stop_freq
    b_notch, a_notch = signal.iirnotch(stop_freq, quality_factor, sig_freq)
    freq, h = signal.freqz(b_notch, a_notch, fs=sig_freq)  # type: ignore
    # plt.plot(freq, 20*np.log10(abs(h)))
    # plt.show()
    x_filtered = signal.filtfilt(b_notch, a_notch, x)
    # plt.plot(x_filtered, 'r-')
    # plt.plot(x, 'b-')
    # plt.show()

    return x_filtered


AUG_PARAMS = {
    'amp_scale_max': 2,
    't_shift_min': -50,
    't_shift_max': 50,
    'dc_shift_min': -10,
    'dc_shift_max': 10,
    'mask_size_min': 0,
    'mask_size_max': 150,
    'mask_pad_min': 0,
    'noise_sd_min': 0,
    'noise_sd_max': 0.2,
    'sig_freq': 500,  # Important to update correctly
    'band_stop_min': 2.8,
    'band_stop_max': 82.5
}


def apply_chosen_aug(x, aug_n, aug_params=None):
    if aug_params is None:
        aug_params = AUG_PARAMS
    match aug_n:
        case 0:
            return amplitude_scale(x, aug_params["amp_shift_max"])
        case 1:
            return time_shift(x, aug_params["time_shift_min"], aug_params["time_shift_max"])
        case 2:
            return dc_shift(x, aug_params["dc_shift_min"], aug_params["dc_shift_max"])
        case 3:
            return zero_mask(x, aug_params["mask_size_min"], aug_params["mask_size_max"], aug_params["mask_pad_min"])
        case 4:
            return add_gaussian_noise(x, aug_params["noise_sd_min"], aug_params["noise_sd_max"])
        case 5:
            return band_stop_filter(x, aug_params["sig_freq"], aug_params["band_stop_min"], aug_params["band_stop_max"])


def create_positive_pair(x, aug_params):
    augs = random.sample(range(6), 2)
    print(augs)



if __name__ == "__main__":

    a = np.random.random((50))
    print(zero_mask(a, 10, 30, 5))



    sample = read_raw_edf(
        "datasets/TUH/normal/01_tcp_ar/aaaaaalk_s002_t000.edf")
    sample.crop(0, 10)

    single_channel = sample.pick_channels(["EEG FP1-REF"])
    sig, times = single_channel[:, :] # type: ignore 
    ch_1 = sig[0:1, :].flatten()
    print(sample.info)
    
    single_copy = single_channel.copy()
    single_copy.rename_channels({'EEG FP1-REF': 'EEG FP1-REF aug'})
    single_copy.load_data()
    single_channel.load_data()
    aug_fn = lambda x: zero_mask(x, 50, 250, 0)  # Tested, works
    # aug_fn = lambda x: time_shift(x, 50, 150)  # Tested, works
    single_copy = single_copy.apply_function(aug_fn, picks=['EEG FP1-REF aug'])
    single_channel.add_channels([single_copy])
    single_channel.plot(block=True)
    # aug_1 = amplitude_scale(ch_1[:], 2)
    # aug_2 = time_shift(ch_1[:], -50, 50)
    aug_bs = band_stop_filter(ch_1[:], sample.info['sfreq'], -2.8, 82.5)


    # fig = .plot(block=True)
