# -*- coding: utf-8 -*-
"""Stimulus resampling functions
"""

import numpy as np
from scipy.fftpack import ifft, fft, ifftshift, fftfreq
from scipy.signal import get_window
import warnings
from distutils.version import LooseVersion
from mne.filter import resample as _resample

from .._parallel import parallel_func, _check_n_jobs


def _fft_resample(x, W, new_len, npad, to_remove):
    """Do FFT resampling with a filter function (possibly using CUDA)

    Parameters
    ----------
    x : 1-d array
        The array to resample.
    W : 1-d array or gpuarray
        The filtering function to apply.
    new_len : int
        The size of the output array (before removing padding).
    npad : int
        Amount of padding to apply before resampling.
    to_remove : int
        Number of samples to remove after resampling.
    cuda_dict : dict
        Dictionary constructed using setup_cuda_multiply_repeated().

    Returns
    -------
    x : 1-d array
        Filtered version of x.
    """
    # add some padding at beginning and end to make this work a little cleaner
    x = _smart_pad(x, npad)
    old_len = len(x)
    N = int(min(new_len, old_len))
    sl_1 = slice((N + 1) // 2)
    y_fft = np.zeros(new_len, np.complex128)
    x_fft = fft(x).ravel()
    x_fft *= W
    y_fft[sl_1] = x_fft[sl_1]
    sl_2 = slice(-(N - 1) // 2, None)
    y_fft[sl_2] = x_fft[sl_2]
    y = np.real(ifft(y_fft, overwrite_x=True)).ravel()

    # now let's trim it back to the correct size (if there was padding)
    if to_remove > 0:
        keep = np.ones((new_len), dtype='bool')
        keep[:to_remove] = False
        keep[-to_remove:] = False
        y = np.compress(keep, y)

    return y


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
