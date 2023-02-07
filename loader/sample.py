from copy import copy, deepcopy

import numpy as np
import matplotlib.pyplot as plt
# from scipy.interpolate import interp1d


class Sample():

    def __init__(self, origin_dataset, channel_names, sr, signal,
                 extra_names=None, extra_srs=None, extra_signals=None):
        self.origin_dataset = origin_dataset  # string
        self.channel_names = channel_names  # List of strings
        self.sr = sr  # int, Hz frequency
        self.signal = signal  # np array
        self.extra_names = extra_names  # List of strings
        self.extra_srs = extra_srs  # int
        self.extra_signals = extra_signals  # np array

        self.channel_indexes = {}  # TODO: find out whether it is necessary, or if a general channel renaming + sorting during dataset loading could be done.
        for i, c in enumerate(channel_names):
            self.channel_indexes[c] = i
        self.seconds = self.signal.shape[1] / self.sr

    def get_signal_slice(self, start=0, stop=None):
        return self.signal[:, start:stop]

    def get_n_channels(self):
        return len(self.channel_names)

    def resampled_copy(self, new_sr):
        new_n_samples = self.signal.shape[1] * new_sr / self.sr

        resampled_sig = np.zeros((self.signal.shape[0], new_n_samples))
        old_t = np.linspace(0, self.seconds, self.signal.shape[1])
        new_t = np.linspace(0, self.seconds, new_n_samples)
        for channel in range(self.get_n_channels()):
            resampled_sig[channel, :] = np.interp(
                new_t, old_t, fp=self.signal[channel, :])

        cpy = Sample(
            self.origin_dataset, copy(
                self.channel_names), new_sr, resampled_sig,
            extra_names=copy(self.extra_names),
            extra_srs=copy(self.extra_srs),
            extra_signals=deepcopy(self.extra_signals)
        )
        return cpy

    def plot_channels(self, channels=None):
        if channels is None:
            channels = self.channel_names
        for channel in channels:
