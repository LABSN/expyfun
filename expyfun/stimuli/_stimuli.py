"""Stimulus generation functions
"""

import warnings
import numpy as np
from scipy.io import wavfile
from scipy import signal
from os import path as op

from .._sound_controllers import SoundPlayer
from .._utils import verbose_dec, logger, _has_scipy_version, wait_secs


def window_edges(sig, fs, dur=0.01, axis=-1, window='hann', edges='both'):
    """Window the edges of a signal (e.g., to prevent "pops")

    Parameters
    ----------
    sig : array-like
        The array to window.
    fs : float
        The sample rate.
    dur : float
        The duration to window on each edge. The default is 0.01 (10 ms).
    axis : int
        The axis to operate over.
    window : str
        The window to use. For a list of valid options, see
        ``scipy.signal.get_window()``.
    edges : str
        Can be ``'leading'``, ``'trailing'``, or ``'both'`` (default).

    Returns
    -------
    windowed_sig : array-like
        The modified array.
    """
    fs = float(fs)
    sig = np.array(sig)  # this will make a copy
    sig_len = sig.shape[axis]
    win_len = int(dur * fs) * 2
    if win_len > sig_len:
        raise RuntimeError('cannot create window of size {0} for signal with '
                           'length {1}'.format(win_len, sig_len))
    win = signal.windows.get_window(window, win_len)
    valid_edges = ('leading', 'trailing', 'both')
    if edges not in valid_edges:
        raise ValueError('edges must be one of {0}, not "{1}"'
                         ''.format(valid_edges, edges))
    # now we can actually do the calculation
    midpt = win_len // 2
    if edges == 'leading':  # eliminate trailing
        win[midpt:] = 1.
    elif edges == 'trailing':  # eliminate leading
        win[:midpt] = 1.
    insert = np.ones(sig_len - win_len, dtype=win.dtype)
    win = np.concatenate((win[:midpt], insert, win[midpt:]))
    shape = np.ones_like(sig.shape)
    shape[axis] = sig.shape[axis]
    win.shape = shape
    sig *= win
    return sig


def rms(data, axis=-1):
    """Calculate the RMS of a signal

    Parameters
    ----------
    data : array-like
        Data to operate on.
    axis : int | None
        Axis to operate over. None will operate over the flattened array.
    """
    return np.sqrt(np.mean(data * data, axis=axis))


def _get_dtype_norm(dtype):
    """Helper to get normalization factor for a given datatype"""
    if np.dtype(dtype).kind == 'i':
        info = np.iinfo(dtype)
        maxval = min(-info.min, info.max)
    else:  # == 'f'
        maxval = 1.0
    return maxval


def _print_wav_info(pre, data, dtype):
    """Helper to print WAV info"""
    logger.info('{0} WAV file with {1} channel{3} and {2} samples '
                '(format {4})'.format(pre, data.shape[0], data.shape[1],
                                      's' if data.shape[0] != 1 else '',
                                      dtype))


@verbose_dec
def read_wav(fname, verbose=None):
    """Read in a WAV file

    Parameters
    ----------
    fname : str
        Filename to load.

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
    format : numpy dtype
        The output format to use. np.int16 is standard for many wav files,
        but np.float32 or np.float64 has higher dynamic range.
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
    _print_wav_info('Writing', data, data.dtype)
    max_val = _get_dtype_norm(dtype)
    data = (data * max_val).astype(dtype)
    wavfile.write(fname, fs, data.T)


def play_sound(sound, fs=44100, norm=True, wait=False):
    """Play a sound

    Parameters
    ----------
    sound : array
        1D or 2D array of sound values.
    fs : int
        Sample rate.
    norm : bool
        If True, normalize sound to between -1 and +1.
    wait : bool
        If True, wait until the sound completes to return control.

    Returns
    -------
    snd : instance of SoundPlayer
        The object playing sound. Can use "stop" to stop playback.
    """
    sound = np.array(sound)
    fs = int(fs)
    if sound.ndim == 1:  # make it stereo
        sound = np.array((sound, sound))
    if sound.ndim != 2:
        raise ValueError('sound must be 1- or 2-dimensional')
    if norm:
        m = np.abs(sound).max() * 1.000001
        m = m if m != 0 else 1
        sound /= m
    if np.abs(sound).max() > 1.:
        warnings.warn('Sound exceeds +/-1, will clip')
    snd = SoundPlayer(sound, fs)
    snd.play()  # will clip as necessary
    if wait:
        wait_secs(sound.shape[1] / float(fs))
    return snd
