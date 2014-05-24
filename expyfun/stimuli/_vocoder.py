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
def vocode(data, fs, n_bands=16, freq_lims=(200., 8000.), scale='erb',
           order=2, env_lp=160., env_lp_order=4, mode='noise',
           poisson_rate=200, poisson_linked=False,
           rand_seed=None, axis=-1, verbose=None):
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
    scale : str
        Scale on which to equally space the bands. Possible values are "erb",
        "loghz" (base-2), and "hz".
    order : int
        Order of analysis and synthesis.
        NOTE: Using too high an order can cause instability,
        always check outputs for order > 2!
    env_lp : float
        Frequency of the envelope low-pass.
    evn_lp_order : int
        Order of the envelope low-pass.
    mode : str
        The type of signal used to excite each band. Options are "noise" for
        band-limited noise, "tone" for sinewave-at-center-frequency, or
        "poisson" for a poisson process of band-limited clicks at the rate
        given by ``poisson_rate``.
    poisson_rate : int
        Average number of clicks per second for the poisson train used to
        excite each band (when mode=="poisson").
    poisson_linked : bool
        Whether the same poisson process is used to excite all bands.
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
    # check args
    if mode not in ('noise', 'tone', 'poisson'):
        raise ValueError('mode must be "noise", "tone", or "poisson", not {0}'
                         ''.format(mode))
    if rand_seed is None:
        rng = np.random
    else:
        rng = np.random.RandomState(rand_seed)
    # get band envelopes
    bands = band_envs(data, fs, n_bands=n_bands, freq_lims=freq_lims,
                      scale=scale, order=order, env_lp=env_lp,
                      env_lp_order=env_lp_order, axis=axis, verbose=verbose)

    n_samp = data.shape[axis]
    voc = np.zeros_like(data)

    # reconstruct
    if poisson_linked and mode == 'poisson':  # same for each band
        carrier = rng.random_integers(0, 1, n_samp).astype(float)
    for cf, env, (b, a) in bands:
        if mode == 'tone':
            carrier = np.sin(2 * np.pi * cf * np.arange(n_samp) / fs)
            carrier *= np.sqrt(2)  # rms of 1
            shape = np.ones_like(data.shape)
            shape[axis] = n_samp
            carrier.shape = shape
        else:
            if mode == 'noise':
                carrier = rng.rand(*data.shape)
            else:  # mode == 'poisson'
                if not poisson_linked:  # different for each band
                    carrier = rng.random_integers(0, 1, n_samp).astype(float)
            carrier = lfilter(b, a, carrier, axis=axis)
            carrier /= np.sqrt(np.mean(carrier * carrier, axis=axis,
                                       keepdims=True))  # rms of 1
        voc += carrier * env
    return voc


@verbose_dec
def band_envs(data, fs, n_bands=16, freq_lims=(200., 8000.), scale='erb',
              order=2, env_lp=160., env_lp_order=4, axis=-1, verbose=None):
    """Calculate frequency band envelopes of a signal

    Parameters
    ----------
    data : array-like
        Data array.
    fs : float
        Sample rate.
    n_bands : int
        Number of bands to use.
    freq_lims : tuple
        2-element list of lower and upper frequency bounds (in Hz).
    scale : str
        Scale on which to equally space the bands. Possible values are "erb",
        "loghz" (base-2), and "hz".
    order : int
        Order of analysis and synthesis.
        NOTE: Using too high an order can cause instability,
        always check outputs for order > 2!
    env_lp : float
        Frequency of the envelope low-pass.
    evn_lp_order : int
        Order of the envelope low-pass.
    axis : int
        Axis to operate over.

    Returns
    -------
    edges, cfs, env : list of tuples
        List of tuples: (center frequency, envelope, (filter numer, denom)).

    Notes
    -----
    Adapted from an algorithm described by Zachary Smith (Cochlear Corp.).
    """

    data = np.array(data, float)  # will make a copy
    freq_lims = np.array(freq_lims, float)
    fs = float(fs)
    if np.any(freq_lims >= fs / 2.) or env_lp >= fs / 2.:
        raise ValueError('frequency limits must not exceed Nyquist')
    assert freq_lims.ndim == 1 and freq_lims.size == 2
    if scale not in ('erb', 'loghz', 'hz'):
        raise ValueError('Frequency scale must be "erb", "hz", or "loghz".')
    if scale == 'erb':
        freq_lims_erbn = _freq_to_erbn(freq_lims)
        delta_erb = np.diff(freq_lims_erbn) / n_bands
        cutoffs = _erbn_to_freq(freq_lims_erbn[0] +
                                delta_erb * np.arange(n_bands + 1))
        assert np.allclose(cutoffs[[0, -1]], freq_lims)  # should be
    elif scale == 'loghz':
        freq_lims_log = np.log2(freq_lims)
        delta = np.diff(freq_lims_log) / n_bands
        cutoffs = 2. ** (freq_lims_log[0] + delta * np.arange(n_bands + 1))
        assert np.allclose(cutoffs[[0, -1]], freq_lims)  # should be
    else:  # scale == 'hz'
        delta = np.diff(freq_lims) / n_bands
        cutoffs = freq_lims[0] + delta * np.arange(n_bands + 1)
    cfs = list(np.round((cutoffs[:-1] + cutoffs[1:]) / 2.).astype(int))
    logger.info('Using frequencies centered at: {0}'.format(cfs))
    # extract the envelope in each band
    envs = []
    filts = []
    for lf, hf in zip(cutoffs[:-1], cutoffs[1:]):
        # band-pass
        b, a = butter(order, [lf / fs, hf / fs], 'bandpass')
        env = lfilter(b, a, data, axis=axis)
        # half-wave rectify and low-pass
        env[env < 0] = 0.0
        b_env, a_env = butter(env_lp_order, env_lp / fs, 'lowpass')
        env = lfilter(b_env, a_env, env, axis=axis)
        envs.append(env)
        filts.append((b, a))
    return(zip(cfs, envs, filts))
