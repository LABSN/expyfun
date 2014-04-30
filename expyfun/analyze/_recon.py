"""Functions for fixing data.
"""

import numpy as np
from scipy import linalg


def restore_values(correct, other, idx):
    """Restore missing values from one sequence using another

    Parameters
    ----------
    correct : array
        1D array of correct values.
    other : array
        1D array of other values that have missing components.
    idx : array
        Indices of the values in correct that are missing from other.
        If None, they will be estimated from the data.

    Returns
    -------
    other : array
        Array of other values, with interpolated values inserted.
    idx : array
        Array of indices that were interpolated.

    Notes
    -----
    This function works even if ``correct`` and ``other`` have different
    scales and shifts. Consider an experiment where the stimulus start
    times were saved by ``ExperimentController`` but the Eyelink system
    missed a ``SYNCTIME`` -- this function allows the proper sample numbers
    to be accurately estimated.
    """
    correct = np.array(correct, np.float64)
    other = np.array(other, np.float64)
    if correct.ndim != 1 or other.ndim != 1 or other.size > correct.size:
        raise RuntimeError('correct and other must be 1D, and correct must '
                           'be at least as long as other')
    keep = np.ones(len(correct), bool)
    for ii in idx:
        keep[ii] = False
    replace = np.where(~keep)[0]
    keep = np.where(keep)[0]
    use = correct[keep]

    X = linalg.pinv(np.array((np.ones_like(use), use)).T)
    X = np.dot(X, other)
    test = np.dot(np.array((np.ones_like(use), use)).T, X)
    if not np.allclose(other, test):  # validate fit
        raise RuntimeError('data could not be fit')
    miss = correct[replace]
    vals = np.dot(np.array((np.ones_like(miss), miss)).T, X)
    out = np.zeros(len(correct), np.float64)
    out[keep] = other
    out[replace] = vals
    return out, replace
