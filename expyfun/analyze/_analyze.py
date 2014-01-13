"""Analysis functions (mostly for psychophysics data).
"""

import warnings
import numpy as np
import scipy.stats as ss


def logit(prop, max_events=None):
    """Convert proportion (expressed in the range [0, 1]) to logit.

    Parameters
    ----------
    pct : float | array-like
        the occurrence proportion.
    max_events : int | None
        the number of events used to calculate ``pct``.  Used in a correction
        factor for cases when ``pct`` is 0 or 1, to prevent returning ``inf``.
        If ``None``, no correction is done, and ``inf`` or ``-inf`` may result.

    Returns
    -------
    numpy.array, with shape matching np.array(prop).shape
    """
    prop = np.asanyarray(prop, dtype=float)
    if np.any([prop > 1, prop < 0]):
        raise ValueError('Proportions must be in the range [0, 1].')
    if max_events is not None:
        # add equivalent of half an event to 0s, and subtract same from 1s
        corr_factor = 0.5 / max_events
        for loc in zip(*np.where(prop == 0)):
            prop[loc] = corr_factor
        for loc in zip(*np.where(prop == 1)):
            prop[loc] = 1 - corr_factor
    return np.log(prop / (np.ones_like(prop) - prop))


def dprime(hmfc, zero_correction=True):
    """Estimates d-prime, with optional correction factor to avoid infinites.

    Parameters
    ----------
    hmfc : array-like
        Hits, misses, false-alarms, and correct-rejections, in that order, as a
        four-element list, tuple, or numpy array.  If an Nx4 array is provided,
        it will return an array of dimension (N,).
    zero_correction : bool
        Whether to add a correction factor of 0.5 to each category to prevent
        division-by-zero leading to infinite d-prime values.

    Notes
    -----
    For two-alternative forced-choice tasks, it is recommended to enter correct
    trials as hits and incorrect trials as false alarms, and enter misses and
    correct rejections as 0. An alternative is to use ``dprime_2afc()``, which
    wraps to ``dprime()`` and does this assignment for you.
    """
    vector = False
    hmfc = _check_dprime_inputs(hmfc)
    if len(hmfc.shape) == 1:
        vector = True
        hmfc = np.atleast_2d(hmfc)
    if zero_correction:
        a = 0.5
    else:
        a = 0.0
    dp = ss.norm.ppf((hmfc[:, 0] + a) / (hmfc[:, 0] + hmfc[:, 1] + 2 * a)) \
        - ss.norm.ppf((hmfc[:, 2] + a) / (hmfc[:, 2] + hmfc[:, 3] + 2 * a))
    if vector:
        return dp[0]
    else:
        return dp


def dprime_2afc(hm, zero_correction=True):
    """Estimates d-prime for two-alternative forced-choice paradigms.

    Parameters
    ----------
    hm : array-like
        Correct trials (hits) and incorrect trials (misses), in that order, as
        a two-element list, tuple, or numpy array. If an Nx2 array is provided,
        it will return an array of dimension (N,).
    zero_correction : bool
        Whether to add a correction factor of 0.5 to each category to prevent
        division-by-zero leading to infinite d-prime values.
    """
    hmfc = _check_dprime_inputs(hm, True)
    return dprime(hmfc, zero_correction)


def _check_dprime_inputs(hmfc, tafc=False):
    """Formats input to dprime() and dprime_2afc().

    Parameters
    ----------
    hmfc : array-like
        Hit, miss, false-alarm, correct-rejection; or hit, miss for 2AFC.
    tafc : bool
        Is this a 2AFC design?
    """
    hmfc = np.array(hmfc)
    if len(hmfc.shape) > 2:
        raise ValueError('Argument to dprime() cannot have more than two '
                         'dimensions.')
    elif hmfc.shape[-1] != 2 and tafc:
        raise ValueError('Array dimensions of argument to dprime_2afc() must '
                         'be (2,) or (N, 2).')
    elif hmfc.shape[-1] != 4 and not tafc:
        raise ValueError('Array dimensions of argument to dprime() must be '
                         '(4,) or (N, 4).')

    if len(hmfc.shape) == 1 and tafc:
        hmfc = np.c_[hmfc[0], 0, hmfc[1], 0]
    elif tafc:
        z = np.zeros_like(hmfc[:, 0])
        hmfc = np.c_[hmfc[:, 0], z, hmfc[:, 1], z]
    if hmfc.dtype not in [np.int64, np.int32]:
        warnings.warn('Argument to dprime() cast to np.int64; floating point '
                      'values will have been truncated.')
        hmfc = hmfc.astype(np.int64)
    return hmfc
