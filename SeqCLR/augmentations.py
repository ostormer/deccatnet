from random import randint, random
import numpy as np
from scipy import signal


def amplitude_scale(x, min_scale, max_scale):
    """Rescale. Equal chance of upscale or downscale,
    scaling factor chosen uniformly, between 1 and max_scale

    Args:
        x (array): 1d signal to rescale
        min_scale (float): UNUSED
        max_scale (float): factor by which to scale signal up or down

    Returns:
        array: scaled 1d signal
    """
    factor = 1 + random()*(max_scale-1)
    if randint(0,1):
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


def zero_mask(x, min_size, max_size, min_pad=0):
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


def band_stop_filter(x, min_freq, max_freq):
    # TODO: Implement this using scipy signal
    # see https://stackoverflow.com/questions/50247517/python-specific-frequency-removenotch-filter/63038706#63038706
    return
