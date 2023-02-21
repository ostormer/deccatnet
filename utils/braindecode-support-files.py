# Author: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD (3-clause)

import tempfile

import numpy as np
import matplotlib.pyplot as plt
import mne

from braindecode.datasets import TUH
from braindecode.preprocessing import (
    preprocess, Preprocessor, create_fixed_length_windows, scale as multiply)


plt.style.use('seaborn')
mne.set_log_level('ERROR')  # avoid messages everytime a window is extracted