import os
import typing

import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from braindecode.datasets.base import BaseConcatDataset

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class ContrastiveAugmentedDataset(BaseConcatDataset):
    """
    BaseConcatDataset is a ConcatDataset from pytorch, which means thath this should be ok.
    """
    def __init__(self, list_of_ds, target_transform=None):
        super().__init__(list_of_ds,target_transform=target_transform)

    def _get_sequence(self, indices):
        X, y = list(), list()
        for ind in indices:
            out_i = super().__getitem__(ind)
            X.append(out_i[0])
            y.append(out_i[1])

        X = np.stack(X, axis=0)
        y = np.array(y)

        return X, y

    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx : int | list
            Index of window and target to return. If provided as a list of
            ints, multiple windows and targets will be extracted and
            concatenated. The target output can be modified on the
            fly by the ``traget_transform`` parameter.
        """
        if isinstance(idx, typing.Iterable):  # Sample multiple windows
            item = self._get_sequence(idx)
        else:
            item = super().__getitem__(idx)
        if self.target_transform is not None:
            item = item[:1] + (self.target_transform(item[1]),) + item[2:]
        return item



