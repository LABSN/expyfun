# -*- coding: utf-8 -*-
"""Vocoder functions
"""

import numpy as np
from scipy.signal import butter, lfilter, filtfilt

from .._utils import verbose_dec


def _freq_to_erbn(f):
    """Convert frequency to ERB number"""
    return 21.4 * np.log10(0.00437 * f + 1)


def _erbn_to_freq(e):
    """Convert ERB number to frequency"""
    return (10 ** (e / 21.4) - 1) / 0.00437


@verbose_dec
def get_band_freqs(fs, n_bands=16, freq_lims=(200., 8000.), scale='erb'):
    """Calculate frequency band edges.

    Parameters
    ----------
    fs : float
        Sample rate.
    n_bands : int
        Number of bands to use.
    freq_lims : tuple
        2-element list of lower and upper frequency bounds (in Hz).
    scale : str
        Scale on which to equally space the bands. Possible values are "erb",
        "log" (base-2), and "hz".

    Returns
    -------
    edges : list of tuples
        low- and high-cutoff frequencies for the bands.
    """
    freq_lims = np.array(freq_lims, float)
    fs = float(fs)
    if np.any(freq_lims >= fs / 2.):
        raise ValueError('frequency limits must not exceed Nyquist')
    assert freq_lims.ndim == 1 and freq_lims.size == 2
    if scale not in ('erb', 'log', 'hz'):
        raise ValueError('Frequency scale must be "erb", "hz", or "log".')
    if scale == 'erb':
        freq_lims_erbn = _freq_to_erbn(freq_lims)
        delta_erb = np.diff(freq_lims_erbn) / n_bands
        cutoffs = _erbn_to_freq(freq_lims_erbn[0] +
                                delta_erb * np.arange(n_bands + 1))
        assert np.allclose(cutoffs[[0, -1]], freq_lims)  # should be
    elif scale == 'log':
        freq_lims_log = np.log2(freq_lims)
        delta = np.diff(freq_lims_log) / n_bands
        cutoffs = 2. ** (freq_lims_log[0] + delta * np.arange(n_bands + 1))
        assert np.allclose(cutoffs[[0, -1]], freq_lims)  # should be
    else:  # scale == 'hz'
        delta = np.diff(freq_lims) / n_bands
        cutoffs = freq_lims[0] + delta * np.arange(n_bands + 1)
    edges = zip(cutoffs[:-1], cutoffs[1:])
    return(edges)


def get_bands(data, fs, edges, order=2, zero_phase=False, axis=-1):
    """Separate a signal into frequency bands

    Parameters
    ----------
    data : array-like
        Data array.
    fs : float
        Sample rate.
    edges : list
        List of tuples of band cutoff frequencies.
    order : int
        Order of analysis and synthesis.
        NOTE: Using too high an order can cause instability,
        always check outputs for order > 2!
    zero_phase : bool
        Use zero-phase forward-backward filtering.
    axis : int
        Axis to operate over.

    Returns
    -------
    bands, filts : list of tuples
        List of tuples (bandpassed signal, (filter coefs numer, denom))
    """
    data = np.atleast_1d(np.array(data, float))  # will make a copy
    fs = float(fs)
    bands = []
    filts = []
    for lf, hf in edges:
        # band-pass
        b, a = butter(order, [lf / fs, hf / fs], 'bandpass')
        filt = filtfilt if zero_phase else lfilter
        band = filt(b, a, data, axis=axis)
        bands.append(band)
        filts.append((b, a))
    return(bands, filts)


def get_env(data, fs, lp_order=4, lp_cutoff=160., zero_phase=False, axis=-1):
    """Calculate a low-pass envelope of a signal

    Parameters
    ----------
    data : array-like
        Data array.
    fs : float
        Sample rate.
    lp_order : int
        Order of the envelope low-pass.
    lp_cutoff : float
        Cutoff frequency of the envelope low-pass.
    zero_phase : bool
        Use zero-phase forward-backward filtering.
    axis : int
        Axis to operate over.

    Returns
    -------
    env, filt : tuple
        Tuple where first element is the rectified and low-pass filtered
        envelope of ``data``, second element is a tuple of the filter
        coefficients (numerator, denominator).
    """
    if lp_cutoff >= fs / 2.:
        raise ValueError('frequency limits must not exceed Nyquist')
    cutoff = lp_cutoff / float(fs)
    data[data < 0] = 0.  # half-wave rectify
    b, a = butter(lp_order, cutoff, 'lowpass')
    filt = filtfilt if zero_phase else lfilter
    env = filt(b, a, data, axis=axis)
    return(env, (b, a))


def get_carriers(data, fs, edges, order=2, axis=-1, mode='tone', rate=None,
                 seed=None):
    """Generate carriers for frequency bands of a signal

    Parameters
    ----------
    data : array-like
        Data array.
    fs : float
        Sample rate.
    edges : list
        List of tuples of band cutoff frequencies.
    order : int
        Order of analysis and synthesis.
        NOTE: Using too high an order can cause instability,
        always check outputs for order > 2!
    axis : int
        Axis to operate over.
    mode : str
        The type of signal used to excite each band. Options are "noise" for
        band-limited noise, "tone" for sinewave-at-center-frequency, or
        "poisson" for a poisson process of band-limited clicks at the rate
        given by ``rate``.
    rate : int
        The mean rate of stimulation when ``mode=='poisson'`` (in clicks per
        second). Ignored when ``mode != 'poisson'``.
    seed : np.random.RandomState | int | None
        Random seed to use. If ``None``, no seeding is done.

    Returns
    -------
    carrs : list of nd-arrays
        List of numpy nd-arrays of the carrier signals.
    """
    # check args
    if mode not in ('noise', 'tone', 'poisson'):
        raise ValueError('mode must be "noise", "tone", or "poisson", not {0}'
                         ''.format(mode))
    if isinstance(seed, np.random.RandomState):
        rng = seed
    elif seed is None:
        rng = np.random
    else:
        try:
            seed = int(seed)
            rng = np.random.RandomState(seed)
        except TypeError:
            raise TypeError('"seed" must be castable to int(), an instance of'
                            ' numpy.random.RandomState, or None.')
            raise

    carrs = []
    fs = float(fs)
    n_samp = data.shape[axis]
    for lf, hf in edges:
        if mode == 'tone':
            cf = (lf + hf) / 2.
            carrier = np.sin(2 * np.pi * cf * np.arange(n_samp) / fs)
            carrier *= np.sqrt(2)  # rms of 1
            shape = np.ones_like(data.shape)
            shape[axis] = n_samp
            carrier.shape = shape
        else:
            if mode == 'noise':
                carrier = rng.rand(*data.shape)
            else:  # mode == 'poisson'
                prob = rate / fs
                carrier = rng.choice([0., 1.], n_samp, p=[1 - prob, prob])
            b, a = butter(order, [lf / fs, hf / fs], 'bandpass')
            carrier = lfilter(b, a, carrier, axis=axis)
            carrier /= np.sqrt(np.mean(carrier * carrier, axis=axis,
                                       keepdims=True))  # rms of 1
        carrs.append(carrier)
    return(carrs)


@verbose_dec
def vocode(data, fs, n_bands=16, freq_lims=(200., 8000.), scale='erb',
           order=2, lp_cutoff=160., lp_order=4, mode='noise',
           rate=200, seed=None, axis=-1, verbose=None):
    """Vocode stimuli using a variety of methods

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
        "log" (base-2), and "hz".
    order : int
        Order of analysis and synthesis.
        NOTE: Using too high an order can cause instability,
        always check outputs for order > 2!
    lp_cutoff : float
        Frequency of the envelope low-pass.
    lp_order : int
        Order of the envelope low-pass.
    mode : str
        The type of signal used to excite each band. Options are "noise" for
        band-limited noise, "tone" for sinewave-at-center-frequency, or
        "poisson" for a poisson process of band-limited clicks at the rate
        given by ``poisson_rate``.
    rate : int
        Average number of clicks per second for the poisson train used to
        excite each band (when mode=="poisson").
    seed : int | None
        Random seed to use. If ``None``, no seeding is done.
    axis : int
        Axis to operate over.

    Returns
    -------
    voc : array-like
        Vocoded stimuli of the same shape as data.

    Notes
    -----
    The default settings are adapted from a cochlear implant simulation
    algorithm described by Zachary Smith (Cochlear Corp.).
    """
    edges = get_band_freqs(fs, n_bands=n_bands, freq_lims=freq_lims,
                           scale=scale)
    bands, filts = get_bands(data, fs, edges, order=order, axis=axis)
    envs, env_filts = zip(*[get_env(x, fs, lp_order=lp_order,
                                    lp_cutoff=lp_cutoff, axis=axis)
                            for x in bands])
    carrs = get_carriers(data, fs, edges, order=order, axis=axis, mode=mode,
                         rate=rate, seed=seed)
    # reconstruct
    voc = np.zeros_like(data)
    for carr, env in zip(carrs, envs):
        voc += carr * env
    return voc
