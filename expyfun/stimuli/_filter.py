# -*- coding: utf-8 -*-
"""Stimulus resampling functions
"""

import numpy as np
from distutils.version import LooseVersion
from mne.filter import resample as _resample


def _smart_pad(x, n_pad):
    """Pad vector x
    """
    # need to pad with zeros if len(x) <= npad
    z_pad = np.zeros(max(n_pad - len(x) + 1, 0), dtype=x.dtype)
    return np.r_[z_pad, 2 * x[0] - x[n_pad:0:-1], x,
                 2 * x[-1] - x[-2:-n_pad - 2:-1], z_pad]


try:
    import mne
    if LooseVersion(mne.__version__) < LooseVersion('0.8'):
        raise ImportError('mne-python too old')
    from mne.filter import resample
except Exception:
    resample = _resample
