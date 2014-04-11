# -*- coding: utf-8 -*-
"""Vocoder functions
"""

import numpy as np
from scipy.signal import butter, lfilter

from .._utils import verbose_dec, logger


def _freq_to_erbn(f):
    """Convert frequency to ERB number"""
    return 21.4 * np.log10(0.00437 * f + 1)


def _erbn_to_freq(e):
    """Convert ERB number to frequency"""
    return (10 ** (e / 21.4) - 1) / 0.00437


@verbose_dec
def vocode_ci(data, fs, n_bands=16, freq_lims=(200., 8000.), order=2,
              env_lp=160., env_lp_order=4, mode='noise', rand_seed=None,
              axis=-1, verbose=None):
    """Vocode stimuli using a CI simulation

    Adapted from an algorithm described by Zachary Smith (Cochlear).

    Parameters
    ----------
    data : array-like
        Data array.
    fs : float
        Sample rate.
    n_bands : int
        Number of bands to use.
    freq_lims : tuple
        2-element list of lower and upper frequency bounds.
    order : int
        Order of analysis and synthesis.
        NOTE: Using too high an order can cause instability,
        always check outputs for order > 2!
    env_lp : float
        Frequency of the envelope low-pass.
    evn_lp_order : int
        Order of the envelope low-pass.
    mode : str
        Use 'noise' or 'tone' for noise- or tone-vocoding.
    rand_seed : int | None
        Random seed to use. If None, no seeding is done.
    axis : int
        Axis to operate over.

    Returns
    -------
    voc : array-like
        Vocoded stimuli of the same shape as data.

    Notes
    -----
    This will use ERB-spaced frequency windows.
    """
    data = np.array(data, float)  # will make a copy
    n_samp = data.shape[axis]
    voc = np.zeros_like(data)
    freq_lims = np.array(freq_lims, float)
    fs = float(fs)
    if np.any(freq_lims >= fs / 2.) or env_lp >= fs / 2.:
        raise ValueError('frequency limits must not exceed Nyquist')
    assert freq_lims.ndim == 1 and freq_lims.size == 2
    if mode not in ('noise', 'tone'):
        raise ValueError('mode must be "noise" or "tone", not {0}'
                         ''.format(mode))
    if rand_seed is None:
        rng = np.random
    else:
        rng = np.random.RandomState(rand_seed)
    freq_lims_erbn = _freq_to_erbn(freq_lims)
    delta_erb = np.diff(freq_lims_erbn) / n_bands
    cutoffs = _erbn_to_freq(freq_lims_erbn[0] +
                            delta_erb * np.arange(n_bands + 1))
    assert np.allclose(cutoffs[[0, -1]], freq_lims)  # should be
    cfs = list(np.round((cutoffs[:-1] + cutoffs[1:]) / 2.).astype(int))
    logger.info('Using frequencies centered at: {0}'.format(cfs))

    # extract the envelope in each band
    for lf, hf in zip(cutoffs[:-1], cutoffs[1:]):
        # band-pass
        b, a = butter(order, [lf / fs, hf / fs], 'bandpass')
        env = lfilter(b, a, data, axis=axis)
        # half-wave rectify and low-pass
        env[env < 0] = 0.0
        b_env, a_env = butter(env_lp_order, env_lp / fs, 'lowpass')
        env = lfilter(b_env, a_env, env, axis=axis)
        # reconstruct
        if mode == 'tone':
            cf = (lf + hf) / 2.
            carrier = np.sin(2 * np.pi * cf * np.arange(n_samp) / fs)
            carrier *= np.sqrt(2)  # rms of 1
            shape = np.ones_like(data.shape)
            shape[axis] = n_samp
            carrier.shape = shape
        else:  # mode == 'noise'
            carrier = rng.rand(*data.shape)
            carrier = lfilter(b, a, carrier, axis=axis)
            carrier /= np.sqrt(np.mean(carrier * carrier, axis=axis,
                                       keepdims=True))  # rms of 1
        voc += carrier * env
    return voc
