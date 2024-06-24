#!/usr/bin/env python2
"""Texture (ERB-spaced) stimulus generation functions."""

# adapted (with permission) from code by Hari Bharadwaj

import warnings

import numpy as np

from .._fixes import irfft
from ._stimuli import rms, window_edges


def _cams(f):
    """Compute cams."""
    return 21.4 * np.log10(0.00437 * f + 1)


def _inv_cams(E):
    """Compute cams inverse."""
    return (10 ** (E / 21.4) - 1.0) / 0.00437


def _scale_sound(x):
    """Scale appropriately to between +/- 1."""
    return 0.95 * x / np.max(np.abs(x))


def _make_narrow_noise(bw, f_c, dur, fs, ramp_dur, rng):
    """Make narrow-band noise using FFT."""
    f_min, f_max = f_c - bw / 2.0, f_c + bw / 2.0
    t = np.arange(int(round(dur * fs))) / fs
    # Make Noise
    f_step = 1.0 / dur  # Frequency bin size
    h_min = int(np.ceil(f_min / f_step))
    h_max = int(np.floor(f_max / f_step)) + 1
    phase = rng.rand(h_max - h_min) * 2 * np.pi
    noise = np.zeros(len(t) // 2 + 1, np.complex128)
    noise[h_min:h_max] = np.exp(1j * phase)
    return window_edges(irfft(noise)[: len(t)], fs, ramp_dur, window="dpss")


def texture_ERB(
    n_freqs=20,
    n_coh=None,
    rho=1.0,
    seq=("inc", "nb", "inc", "nb"),
    fs=24414.0625,
    dur=1.0,
    SAM_freq=7.0,
    random_state=None,
    freq_lims=(200, 8000),
    verbose=True,
):
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
        Sequence of incoherent ('inc'), coherent noise envelope ('nb'), and
        SAM ('sam') mixtures. Default is ``('inc', 'nb', 'inc', 'nb')``.
    fs : float
        Sampling rate in Hz.
    dur : float
        Duration (in seconds) of each token in seq (default is 1.0).
    SAM_freq : float
        The SAM frequency to use.
    random_state : None | int | np.random.RandomState
        The random generator state used for band selection and noise
        envelope generation.
    freq_lims : tuple
        The lower and upper frequency limits (default is (200, 8000)).
    verbose : bool
        If True, print the resulting ERB spacing.

    Returns
    -------
    x : ndarray, shape (n_samples,)
        The stimulus, where ``n_samples = len(seq) * (fs * dur)``
        (approximately).

    Notes
    -----
    This function requires MNE.
    """
    from mne.time_frequency.multitaper import dpss_windows
    from mne.utils import check_random_state

    if not isinstance(seq, (list, tuple, np.ndarray)):
        raise TypeError("seq must be list, tuple, or ndarray, got %s" % type(seq))
    known_seqs = ("inc", "nb", "sam")
    for si, s in enumerate(seq):
        if s not in known_seqs:
            raise ValueError(
                "all entries in seq must be one of %s, got "
                "seq[%s]=%s" % (known_seqs, si, s)
            )
    fs = float(fs)
    rng = check_random_state(random_state)
    n_coh = int(np.round(n_freqs * 0.8)) if n_coh is None else n_coh
    rise = 0.002
    t = np.arange(int(round(dur * fs))) / fs

    f_min, f_max = freq_lims
    n_ERBs = _cams(f_max) - _cams(f_min)
    del f_max
    spacing_ERBs = n_ERBs / float(n_freqs - 1)
    if verbose:
        print(
            "This stim will have successive tones separated by %2.2f ERBs"
            % spacing_ERBs
        )
    if spacing_ERBs < 1.0:
        warnings.warn("The spacing between tones is LESS THAN 1 ERB!")

    # Make a filter whose impulse response is purely positive (to avoid phase
    # jumps) so that the filtered envelope is purely positive. Use a DPSS
    # window to minimize sidebands. For a bandwidth of bw, to get the shortest
    # filterlength, we need to restrict time-bandwidth product to a minimum.
    # Thus we need a length*bw = 2 => length = 2/bw (second). Hence filter
    # coefficients are calculated as follows:
    b = dpss_windows(int(np.floor(2 * fs / 100.0)), 1.0, 1)[0][0]
    b -= b[0]
    b /= b.sum()

    # Incoherent
    envrate = 14
    bw = 20
    incoh = 0.0
    for k in range(n_freqs):
        f = _inv_cams(_cams(f_min) + spacing_ERBs * k)
        env = _make_narrow_noise(bw, envrate, dur, fs, rise, rng)
        env[env < 0] = 0
        env = np.convolve(b, env)[: len(t)]
        incoh += _scale_sound(
            window_edges(env * np.sin(2 * np.pi * f * t), fs, rise, window="dpss")
        )
    incoh /= rms(incoh)

    # Coherent (noise band)
    stims = dict(inc=0.0, nb=0.0, sam=0.0)
    group = np.sort(rng.permutation(np.arange(n_freqs))[:n_coh])
    for kind in known_seqs:
        if kind == "nb":  # noise band
            env_coh = _make_narrow_noise(bw, envrate, dur, fs, rise, rng)
        else:  # 'nb' or 'inc'
            env_coh = 0.5 + np.sin(2 * np.pi * SAM_freq * t) / 2.0
            env_coh = window_edges(env_coh, fs, rise, window="dpss")
        env_coh[env_coh < 0] = 0
        env_coh = np.convolve(b, env_coh)[: len(t)]
        if kind == "inc":
            use_group = []  # no coherent ones
        else:  # 'nb' or 'sam'
            use_group = group
        for k in range(n_freqs):
            f = _inv_cams(_cams(f_min) + spacing_ERBs * k)
            env_inc = _make_narrow_noise(bw, envrate, dur, fs, rise, rng)
            env_inc[env_inc < 0] = 0.0
            env_inc = np.convolve(b, env_inc)[: len(t)]
            if k in use_group:
                env = np.sqrt(rho) * env_coh + np.sqrt(1 - rho**2) * env_inc
            else:
                env = env_inc
            stims[kind] += _scale_sound(
                window_edges(env * np.sin(2 * np.pi * f * t), fs, rise, window="dpss")
            )
        stims[kind] /= rms(stims[kind])
    stim = np.concatenate([stims[s] for s in seq])
    stim = 0.01 * stim / rms(stim)
    return stim
