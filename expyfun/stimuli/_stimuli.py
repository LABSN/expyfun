# -*- coding: utf-8 -*-
"""Generic stimulus generation functions."""

import warnings
import numpy as np
from scipy import signal
from threading import Timer

from ..io import read_wav
from .._sound_controllers import SoundPlayer
from .._utils import wait_secs, string_types


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
        ``scipy.signal.get_window()``, but can also be 'dpss'.
    edges : str
        Can be ``'leading'``, ``'trailing'``, or ``'both'`` (default).

    Returns
    -------
    windowed_sig : array-like
        The modified array (float64).
    """
    fs = float(fs)
    sig = np.array(sig, dtype=np.float64)  # this will make a copy
    sig_len = sig.shape[axis]
    win_len = int(dur * fs)
    if win_len > sig_len:
        raise RuntimeError('cannot create window of size {0} samples (dur={1})'
                           'for signal with length {2}'
                           ''.format(win_len, dur, sig_len))
    if window == 'dpss':
        from mne.time_frequency.multitaper import dpss_windows
        win = dpss_windows(2 * win_len + 1, 1, 1)[0][0][:win_len]
        win -= win[0]
        win /= win.max()
    else:
        win = signal.windows.get_window(window, 2 * win_len)[:win_len]
    valid_edges = ('leading', 'trailing', 'both')
    if edges not in valid_edges:
        raise ValueError('edges must be one of {0}, not "{1}"'
                         ''.format(valid_edges, edges))
    # now we can actually do the calculation
    flattop = np.ones(sig_len, dtype=np.float64)
    if edges in ('trailing', 'both'):  # eliminate trailing
        flattop[-win_len:] *= win[::-1]
    if edges in ('leading', 'both'):  # eliminate leading
        flattop[:win_len] *= win
    shape = np.ones_like(sig.shape)
    shape[axis] = sig.shape[axis]
    flattop.shape = shape
    sig *= flattop
    return sig


def rms(data, axis=-1, keepdims=False):
    """Calculate the RMS of a signal

    Parameters
    ----------
    data : array-like
        Data to operate on.
    axis : int | None
        Axis to operate over. None will operate over the flattened array.
    keepdims : bool
        Keep dimension operated over.
    """
    return np.sqrt(np.mean(data * data, axis=axis, keepdims=keepdims))


def play_sound(sound, fs=None, norm=True, wait=False):
    """Play a sound

    Parameters
    ----------
    sound : array
        1D or 2D array of sound values.
    fs : int | None
        Sample rate. If None, the sample rate will be inferred from the sound
        file (if sound is array, it is assumed to be 44100).
    norm : bool
        If True, normalize sound to between -1 and +1.
    wait : bool
        If True, wait until the sound completes to return control.

    Returns
    -------
    snd : instance of SoundPlayer
        The object playing sound. Can use "stop" to stop playback. Note that
        the sound player will be cleared/deleted once the sound finishes
        playing.
    """
    sound = np.array(sound)
    fs_in = 44100
    if isinstance(sound, string_types):
        sound, fs_in = read_wav(sound)
    if fs is None:
        fs = fs_in
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
    dur = sound.shape[1] / float(fs)
    snd.play()  # will clip as necessary
    del_wait = 0.5
    if wait:
        wait_secs(dur)
    else:
        del_wait += dur
    if hasattr(snd, 'delete'):  # for backward compatibility
        Timer(del_wait, snd.delete).start()
    return snd
