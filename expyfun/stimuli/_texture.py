#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Texture (ERB-spaced) stimulus generation functions."""

# adapted (with permission) from code by Hari Bharadwaj

import numpy as np
import warnings

from ._stimuli import rms, window_edges


def _cams(f):
    """Compute cams."""
    return 21.4 * np.log10(0.00437 * f + 1)


def _inv_cams(E):
    """Compute cams inverse."""
    return (10 ** (E / 21.4) - 1.) / 0.00437


def _scale_sound(x):
    """Scale appropriately to between +/- 1."""
    return 0.95 * x / np.max(np.abs(x))


def _make_narrow_noise(bw, f_c, dur, fs, ramp_dur, rng):
    """Make narrow-band noise using FFT."""
    f_min, f_max = f_c - bw / 2., f_c + bw / 2.
    t = np.arange(int(round(dur * fs))) / fs
    # Make Noise
    f_step = 1. / dur  # Frequency bin size
    h_min = int(np.ceil(f_min / f_step))
    h_max = int(np.floor(f_max / f_step)) + 1
    phase = rng.rand(h_max - h_min) * 2 * np.pi
    noise = np.zeros(len(t) // 2 + 1, np.complex)
    noise[h_min:h_max] = np.exp(1j * phase)
    return window_edges(np.fft.irfft(noise)[:len(t)], fs, ramp_dur,
                        window='dpss')


def texture_ERB(n_freqs=20, n_coh=None, rho=1., seq=(0, 1, 0, 1),
                fs=24414.0625, dur=1., random_state=None):
    """Create ERB texture stimulus

    Parameters
    ----------
    n_freqs : int
        Number of tones in mixture (default 20).
    n_coh : int | None
        Number of tones to be temporally coherent. Default (None) is
        ``int(np.round(n_freqs * 0.8))``.
    rho : float
        Correlation between the envelopes of grouped tones (default is 1.0).
    seq : list
        Sequence of incoherent (False) and ccoherent (True) mixtures.
        Default is ``(0, 1, 0, 1)``.
    fs : float
        Sampling rate in Hz.
    dur : float
        Duration (in seconds) of each token in seq (default is 1.0).

    Returns
    -------
    x : ndarray, shape (N,)
        The stimulus.

    Notes
    -----
    This function requires MNE.
    """
    from mne.time_frequency.multitaper import dpss_windows
    from mne.utils import check_random_state
    fs = float(fs)
    rng = check_random_state(random_state)
    n_coh = int(np.round(n_freqs * 0.8)) if n_coh is None else n_coh
    rise = 0.002
    t = np.arange(int(round(dur * fs))) / fs

    f_min, f_max = 200, 8000
    n_ERBs = _cams(f_max) - _cams(f_min)
    del f_max
    spacing_ERBs = n_ERBs / float(n_freqs - 1)
    print('This stim will have successive tones separated by %2.2f ERBs'
          % spacing_ERBs)
    if spacing_ERBs < 1.0:
        warnings.warn('The spacing between tones is LESS THAN 1 ERB!')

    # Make a filter whose impulse response is purely positive (to avoid phase
    # jumps) so that the filtered envelope is purely positive. Use a DPSS
    # window to minimize sidebands. For a bandwidth of bw, to get the shortest
    # filterlength, we need to restrict time-bandwidth product to a minimum.
    # Thus we need a length*bw = 2 => length = 2/bw (second). Hence filter
    # coefficients are calculated as follows:
    b = dpss_windows(int(np.floor(2 * fs / 100.)), 1., 1)[0][0]
    b -= b[0]
    b /= b.sum()

    # Incoherent
    envrate = 14
    bw = 10
    z = 0.
    for k in range(n_freqs):
        f = _inv_cams(_cams(f_min) + spacing_ERBs * k)
        env = _make_narrow_noise(bw, envrate, dur, fs, rise, rng)
        env[env < 0] = 0
        env = np.convolve(b, env)[:len(t)]
        z += _scale_sound(window_edges(
            env * np.sin(2 * np.pi * f * t), fs, rise, window='dpss'))
    z /= rms(z)

    # Coherent
    group = rng.permutation(np.arange(n_freqs))
    env1 = _make_narrow_noise(bw, envrate, dur, fs, rise, rng)
    env1[env1 < 0] = 0
    env1 = np.convolve(b, env)[:len(t)]
    y = 0
    for k in group[:n_coh]:
        f = _inv_cams(_cams(f_min) + spacing_ERBs * k)
        env2 = _make_narrow_noise(bw, envrate, dur, fs, rise, rng)
        env2[env2 < 0] = 0.
        env2 = np.convolve(b, env2)[:len(t)]
        env = np.sqrt(rho) * env1 + np.sqrt(1 - rho ** 2) * env2
        y += _scale_sound(window_edges(
            env * np.sin(2 * np.pi * f * t), fs, rise, window='dpss'))
    for k in group[n_coh:]:
        f = _inv_cams(_cams(f_min) + spacing_ERBs * k)
        env = _make_narrow_noise(bw, envrate, dur, fs, rise, rng)
        env[env < 0] = 0.
        env = np.convolve(b, env)[:len(t)]
        y += _scale_sound(window_edges(
            env * np.sin(2 * np.pi * f * t), fs, rise, window='dpss'))
    y /= rms(y)

    stim = np.concatenate([y if s else z for s in seq])
    stim = 0.01 * stim / rms(stim)
    return stim
