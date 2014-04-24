"""Analysis functions (mostly for psychophysics data).
"""

import warnings
import numpy as np
import scipy.stats as ss
from scipy.optimize import curve_fit
from functools import partial


def logit(prop, max_events=None):
    """Convert proportion (expressed in the range [0, 1]) to logit.

    Parameters
    ----------
    prop : float | array-like
        the occurrence proportion.
    max_events : int | array-like | None
        the number of events used to calculate ``prop``. Used in a correction
        factor for cases when ``prop`` is 0 or 1, to prevent returning ``inf``.
        If ``None``, no correction is done, and ``inf`` or ``-inf`` may result.

    Returns
    -------
    lgt : ``numpy.ndarray``, with shape matching ``numpy.array(prop).shape``.
    """
    prop = np.atleast_1d(prop).astype(float)
    if np.any([prop > 1, prop < 0]):
        raise ValueError('Proportions must be in the range [0, 1].')
    if max_events is not None:
        # add equivalent of half an event to 0s, and subtract same from 1s
        max_events = np.atleast_1d(max_events) * np.ones_like(prop)
        corr_factor = 0.5 / max_events
        for loc in zip(*np.where(prop == 0)):
            prop[loc] = corr_factor[loc]
        for loc in zip(*np.where(prop == 1)):
            prop[loc] = 1 - corr_factor[loc]
    return np.log(prop / (np.ones_like(prop) - prop))


def sigmoid(x, lower=0., upper=1., midpt=0., slope=1.):
    """Calculate sigmoidal values along the x-axis

    Parameters
    ----------
    x : array-like
        x-values to calculate the sigmoidal values from.
    lower : float
        The lower y-asymptote.
    upper : float
        The upper y-asymptote.
    midpt : float
        The x-value that obtains 50% between the lower and upper asymptote.
    slope : float
        The slope of the sigmoid.

    Returns
    -------
    y : array
        The y-values of the sigmoid evaluated at x.
    """
    x = np.asarray(x)
    lower = float(lower)
    upper = float(upper)
    midpt = float(midpt)
    slope = float(slope)
    y = (upper - lower) / (1 + np.exp(-slope * (x - midpt))) + lower
    return y


def fit_sigmoid(x, y, p0=None):
    """Fit a sigmoid to the data

    Parameters
    ----------
    x : array-like
        x-values along the sigmoid.
    y : array-like
        y-values along the sigmoid.
    p0 : array-like | None
        Initial guesses for the fit. Can be None to have these automatically
        estimated.

    Returns
    -------
    lower, upper, midpt, slope : floats
        See expyfun.analyze.sigmoid for descriptions.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    k = 2 * 4. / (np.max(x) - np.min(x))
    if p0 is None:
        p0 = [np.min(y), np.max(y), np.mean([np.max(x), np.min(x)]), k]
    p0 = np.array(p0, dtype=np.float64)
    if p0.size != 4:
        raise ValueError('p0 must have 4 elements, or be None')
    out = curve_fit(sigmoid, x, y, p0=p0, )[0]
    return out


def rt_chisq(x, axis=None):
    """Chi square fit for reaction times (a better summary statistic than mean)

    Parameters
    ----------
    x : array-like
        Reaction time data to fit.

    axis : int | None
        The axis along which to calculate the chi-square fit. If none, ``x``
        will be flattened before fitting.

    Returns
    -------
    peak : float | array-like
        The peak(s) of the fitted chi-square probability density function(s).

    Notes
    -----
    Verify that it worked by plotting pdf vs hist (for 1-dimensional x)::

        >>> import numpy as np
        >>> from scipy import stats as ss
        >>> import matplotlib.pyplot as plt
        >>> plt.ion()
        >>> x = np.abs(np.random.randn(10000) + 1)
        >>> lsp = np.linspace(np.floor(np.amin(x)), np.ceil(np.amax(x)), 100)
        >>> df, loc, scale = ss.chi2.fit(x, floc=0)
        >>> pdf = ss.chi2.pdf(lsp, df, scale=scale)
        >>> plt.plot(lsp, pdf)
        >>> plt.hist(x, normed=True)
    """
    if np.any(np.less(x, 0)):  # save the user some pain
        raise ValueError('x cannot have negative values')
    if axis is None:
        df, _, scale = ss.chi2.fit(x, floc=0)
    else:
        fit = partial(ss.chi2.fit, floc=0)
        params = np.apply_along_axis(fit, axis=axis, arr=x)  # df, loc, scale
        pmut = np.concatenate((np.atleast_1d(axis),
                               np.delete(np.arange(x.ndim), axis)))
        df = np.transpose(params, pmut)[0]
        scale = np.transpose(params, pmut)[2]
    peak = np.maximum(0, (df - 2)) * scale
    return peak


def dprime(hmfc, zero_correction=True):
    """Estimates d-prime, with optional correction factor to avoid infinites.

    Parameters
    ----------
    hmfc : array-like
        Hits, misses, false-alarms, and correct-rejections, in that order, as a
        four-element list, tuple, or numpy array, or an Nx4 array.
    zero_correction : bool
        Whether to add a correction factor of 0.5 to each category to prevent
        division-by-zero leading to infinite d-prime values.

    Returns
    -------
    dp : float | array
        If ``hmfc`` is a four-element list, tuple, or array, returns a single
        float value. If ``hmfc`` is an Nx4 array, returns an array of dimension
        (N,).

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

    Returns
    -------
    dp : float | array
        If ``hm`` is a two-element list, tuple, or array, returns a single
        float value. If ``hm`` is an Nx2 array, returns an array of dimension
        (N,).
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
        warnings.warn('Argument to dprime() cast to np.int64; floating '
                      'point values will have been truncated.')
        hmfc = hmfc.astype(np.int64)
    return hmfc
