"""Generic stimulus generation functions."""

import warnings
from threading import Timer

import numpy as np
from scipy import signal

from .._sound_controllers import SoundPlayer
from .._utils import _wait_secs
from ..io import read_wav


def window_edges(sig, fs, dur=0.01, axis=-1, window="hann", edges="both"):
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
        raise RuntimeError(
            f"cannot create window of size {win_len} samples (dur={dur})"
            f"for signal with length {sig_len}"
            ""
        )
    if window == "dpss":
        from mne.time_frequency.multitaper import dpss_windows

        win = dpss_windows(2 * win_len + 1, 1, 1)[0][0][:win_len]
        win -= win[0]
        win /= win.max()
    else:
        win = signal.windows.get_window(window, 2 * win_len)[:win_len]
    valid_edges = ("leading", "trailing", "both")
    if edges not in valid_edges:
        raise ValueError(f'edges must be one of {valid_edges}, not "{edges}"')
    # now we can actually do the calculation
    flattop = np.ones(sig_len, dtype=np.float64)
    if edges in ("trailing", "both"):  # eliminate trailing
        flattop[-win_len:] *= win[::-1]
    if edges in ("leading", "both"):  # eliminate leading
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


def play_sound(sound, fs=None, norm=True, wait=False, backend="auto"):
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
    backend : str
        The backend to use.

    Returns
    -------
    snd : instance of SoundPlayer
        The object playing sound. Can use "stop" to stop playback. Note that
        the sound player will be cleared/deleted once the sound finishes
        playing.
    """
    sound = np.array(sound)
    fs_default = 44100
    if isinstance(sound, str):
        sound, fs_default = read_wav(sound)
    if fs is None:
        fs = fs_default
    if sound.ndim == 1:  # make it stereo
        sound = np.array((sound, sound))
    if sound.ndim != 2:
        raise ValueError("sound must be 1- or 2-dimensional")
    if norm:
        m = np.abs(sound).max() * 1.000001
        m = m if m != 0 else 1
        sound /= m
    if np.abs(sound).max() > 1.0:
        warnings.warn("Sound exceeds +/-1, will clip")
    # For rtmixer it's possible this will fail on some configurations if
    # resampling isn't built in to the backend; when we hit this we can
    # try/except here and do the resampling ourselves.
    snd = SoundPlayer(sound, fs=fs, backend=backend)
    dur = sound.shape[1] / float(fs)
    snd.play()  # will clip as necessary
    del_wait = 0.5
    if wait:
        _wait_secs(dur)
    else:
        del_wait += dur
    if hasattr(snd, "delete"):  # for backward compatibility
        Timer(del_wait, snd.delete).start()
    return snd


def add_pad(sounds, alignment="start"):
    """Add sounds of different lengths and channel counts together

    Parameters
    ----------
    sounds : list
        The sounds to add together. There is a maximum of two channels.
    alignment : str
        How to align the sounds. Either by ``'start'`` (add zeros at end),
        ``'center'`` (add zeros on both sides), or ``'end'`` (add zeros at
        start).

    Returns
    -------
    y : float
        The summed sounds. The number of channels will be equal to the maximum
        number of channels (by appending a copy), and the length will be equal
        to the maximum length (by appending zeros).

    Notes
    -----
        Even if the original sounds were all 0- or 1-dimensional, the output
        will be 2-dimensional (channels, samples).
    """
    if alignment not in ["start", "center", "end"]:
        raise ValueError("alignment must be either 'start', 'center', or 'end'")
    x = [np.atleast_2d(y) for y in sounds]
    if not np.all(y.ndim == 2 for y in x):
        raise ValueError("Sound data must have no more than 2 dimensions.")
    shapes = [y.shape for y in x]
    ch_max, len_max = np.max(shapes, axis=0)
    if ch_max > 2:
        raise ValueError("Only 1- and 2-channel sounds are supported.")
    for xi, (ch, length) in enumerate(shapes):
        if length < len_max:
            if alignment == "start":
                n_pre = 0
                n_post = len_max - length
            elif alignment == "center":
                n_pre = (len_max - length) // 2
                n_post = len_max - length - n_pre
            elif alignment == "end":
                n_pre = len_max - length
                n_post = 0
            x[xi] = np.pad(x[xi], ((0, 0), (n_pre, n_post)), "constant")
        if ch < ch_max:
            x[xi] = np.tile(x[xi], [ch_max, 1])
    return np.sum(x, 0)
