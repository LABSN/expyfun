"""Analysis functions (mostly for psychophysics data).
"""

import scipy.stats as ss
import warnings
import numbers

def dprime(h, m, fa, cr, zero_correction=True):
    """Estimates d-prime, with optional correction factor to avoid infinites.

    Parameters
    ----------
    h : int
        Number of detected targets (hits).
    m : int
        Number of undetected targets (misses).
    fa : int
        Number of incorrect non-target detections (false alarms).
    cr : int
        Number of correct non-target detections (correct rejections).
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
    for x in [h, m, fa, cr]:
        if not isinstance(x, numbers.Number):
            raise TypeError('Arguments to d-prime must be numeric.')
        elif not isinstance(x, int):
            warnings.warn('Non-integer arguments to d-prime were truncated.')

    if zero_correction:
        a = 0.5
    else:
        a = 0.0
    return ss.norm.ppf((int(h) + a) / (int(h) + int(m) + 2 * a)) - \
        ss.norm.ppf((int(fa) + a) / (int(fa) + int(cr) + 2 * a))


def dprime_2afc(h, m, zero_correction=True):
    """Estimates d-prime for two-alternative forced-choice paradigms.

    Parameters
    ----------
    h : int
        Number of correct trials (hits).
    m : int
        Number of incorrect trials (misses).
    zero_correction : bool
        Whether to add a correction factor of 0.5 to each category to prevent
        division-by-zero leading to infinite d-prime values.
    """
    return dprime(h, 0, m, 0, zero_correction)
