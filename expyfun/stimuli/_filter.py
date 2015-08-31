# -*- coding: utf-8 -*-
"""Stimulus resampling functions
"""


def resample(x, up, down, npad=100, axis=-1, window='boxcar', n_jobs=1,
             verbose=None):
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
        Number of jobs to run in parallel. Can be 'cuda' if scikits.cuda
        is installed properly and CUDA is initialized.
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
    from mne.filter import resample as _resample
    return _resample(x, up, down, npad, axis, window, n_jobs, verbose)
