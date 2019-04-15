# -*- coding: utf-8 -*-
"""WAV file IO functions
"""

import numpy as np
from scipy.io import wavfile
from os import path as op
import warnings

from .._utils import verbose_dec, logger, _has_scipy_version


@verbose_dec
def read_wav(fname, verbose=None):
    """Read in a WAV file

    Parameters
    ----------
    fname : str
        Filename to load.
    verbose : bool, str, int, or None
        If not None, override default verbose level.

    Returns
    -------
    data : array
        The WAV file data. Will be of datatype np.float64. If the data
        had been saved as integers (typical), this function will
        automatically rescale the data to be between -1 and +1.
        The result will have dimension n_channels x n_samples.
    fs : int
        The wav sample rate
    """
    fs, data = wavfile.read(fname)
    data = np.atleast_2d(data.T)
    orig_dtype = data.dtype
    max_val = _get_dtype_norm(orig_dtype)
    data = np.ascontiguousarray(data.astype(np.float64) / max_val)
    _print_wav_info('Read', data, orig_dtype)
    return data, fs


@verbose_dec
def write_wav(fname, data, fs, dtype=np.int16, overwrite=False, verbose=None):
    """Write a WAV file

    Parameters
    ----------
    fname : str
        Filename to save as.
    data : array
        The data to save.
    fs : int
        The sample rate of the data.
    dtype : numpy dtype
        The output format to use. np.int16 is standard for many wav files,
        but np.float32 or np.float64 has higher dynamic range.
    overwrite : bool
        If True, overwrite the file if necessary.
    verbose : bool, str, int, or None
        If not None, override default verbose level.
    """
    if not overwrite and op.isfile(fname):
        raise IOError('File {} exists, overwrite=True must be '
                      'used'.format(op.basename(fname)))
    if not np.dtype(type(fs)).kind == 'i':
        fs = int(fs)
        warnings.warn('Warning: sampling rate is being cast to integer and '
                      'may be truncated.')
    data = np.atleast_2d(data)
    if np.dtype(dtype).kind not in ['i', 'f']:
        raise TypeError('dtype must be integer or float')
    if np.dtype(dtype).kind == 'f':
        if not _has_scipy_version('0.13'):
            raise RuntimeError('cannot write float datatype unless '
                               'scipy >= 0.13 is installed')
    elif np.dtype(dtype).itemsize == 8:
        raise RuntimeError('Writing 64-bit integers is not supported')
    if np.dtype(data.dtype).kind == 'f':
        if np.dtype(dtype).kind == 'i' and np.max(np.abs(data)) > 1.:
            raise ValueError('Data must be between -1 and +1 when saving '
                             'with an integer dtype')
    _print_wav_info('Writing', data, dtype)
    max_val = _get_dtype_norm(dtype)
    data = (data * max_val).astype(dtype)
    wavfile.write(fname, fs, data.T)


def _print_wav_info(pre, data, dtype):
    """Helper to print WAV info"""
    logger.info('{0} WAV file with {1} channel{3} and {2} samples '
                '(format {4})'.format(pre, data.shape[0], data.shape[1],
                                      's' if data.shape[0] != 1 else '',
                                      dtype))


def _get_dtype_norm(dtype):
    """Helper to get normalization factor for a given datatype"""
    if np.dtype(dtype).kind == 'i':
        info = np.iinfo(dtype)
        maxval = min(-info.min, info.max)
    else:  # == 'f'
        maxval = 1.0
    return maxval
