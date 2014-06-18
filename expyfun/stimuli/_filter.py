# -*- coding: utf-8 -*-
"""Stimulus resampling functions
"""

import numpy as np
from scipy.fftpack import ifft, fft, ifftshift, fftfreq
from scipy.signal import get_window
import warnings
from distutils.version import LooseVersion

from .._parallel import parallel_func, _check_n_jobs


##############################################################################
# RESAMPLE (adapted from mne-python with permission)

def _resample(x, up, down, npad=100, axis=-1, window='boxcar', n_jobs=1):
    """Resample the array x

    Operates along the last dimension of the array.

    Parameters
    ----------
    x : n-d array
        Signal to resample.
    up : float
        Factor to upsample by.
    down : float
        Factor to downsample by.
    npad : integer
        Number of samples to use at the beginning and end for padding.
    axis : int
        Axis along which to resample (default is the last axis).
    window : string or tuple
        See scipy.signal.resample for description.
    n_jobs : int | str
        Number of jobs to run in parallel.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    xf : array
        x resampled.

    Notes
    -----
    This uses (hopefully) intelligent edge padding and frequency-domain
    windowing improve scipy.signal.resample's resampling method, which
    we have adapted for our use here. Choices of npad and window have
    important consequences, and the default choices should work well
    for most natural signals.

    Resampling arguments are broken into "up" and "down" components for future
    compatibility in case we decide to use an upfirdn implementation. The
    current implementation is functionally equivalent to passing
    up=up/down and down=1.
    """
    # check explicitly for backwards compatibility
    if not isinstance(axis, int):
        err = ("The axis parameter needs to be an integer (got %s). "
               "The axis parameter was missing from this function for a "
               "period of time, you might be intending to specify the "
               "subsequent window parameter." % repr(axis))
        raise TypeError(err)

    # make sure our arithmetic will work
    ratio = float(up) / down
    if axis < 0:
        axis = x.ndim + axis
    orig_last_axis = x.ndim - 1
    if axis != orig_last_axis:
        x = x.swapaxes(axis, orig_last_axis)
    orig_shape = x.shape
    x_len = orig_shape[-1]
    if x_len == 0:
        warnings.warn('x has zero length along last axis, returning a copy of '
                      'x')
        return x.copy()

    # prep for resampling now
    x_flat = x.reshape((-1, x_len))
    orig_len = x_len + 2 * npad  # length after padding
    new_len = int(round(ratio * orig_len))  # length after resampling
    to_remove = np.round(ratio * npad).astype(int)

    # figure out windowing function
    if window is not None:
        if callable(window):
            W = window(fftfreq(orig_len))
        elif isinstance(window, np.ndarray) and \
                window.shape == (orig_len,):
            W = window
        else:
            W = ifftshift(get_window(window, orig_len))
    else:
        W = np.ones(orig_len)
    W *= (float(new_len) / float(orig_len))
    W = W.astype(np.complex128)

    # do the resampling using an adaptation of scipy's FFT-based resample()
    # use of the 'flat' window is recommended for minimal ringing
    if n_jobs == 1:
        y = np.zeros((len(x_flat), new_len - 2 * to_remove), dtype=x.dtype)
        for xi, x_ in enumerate(x_flat):
            y[xi] = _fft_resample(x_, W, new_len, npad, to_remove)
    else:
        _check_n_jobs(n_jobs)
        parallel, p_fun, _ = parallel_func(_fft_resample, n_jobs)
        y = parallel(p_fun(x_, W, new_len, npad, to_remove)
                     for x_ in x_flat)
        y = np.array(y)

    # Restore the original array shape (modified for resampling)
    y.shape = orig_shape[:-1] + (y.shape[1],)
    if axis != orig_last_axis:
        y = y.swapaxes(axis, orig_last_axis)

    return y


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
except ImportError:
    resample = _resample
