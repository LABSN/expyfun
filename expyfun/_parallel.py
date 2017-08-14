# -*- coding: utf-8 -*-
"""Parallel util functions
"""

# Adapted from mne-python with permission


def parallel_func(func, n_jobs):
    """Return parallel instance with delayed function

    Util function to use joblib only if available

    Parameters
    ----------
    func: callable
        A function
    n_jobs: int
        Number of jobs to run in parallel

    Returns
    -------
    parallel: instance of joblib.Parallel or list
        The parallel object
    my_func: callable
        func if not parallel or delayed(func)
    n_jobs: int
        Number of jobs >= 0
    """
    # let it error out if the user tries to do parallel w/o joblib
    if n_jobs > 1:
        try:
            from joblib import Parallel, delayed
            # create keyword arguments for Parallel
            n_jobs = _check_n_jobs(n_jobs)
            parallel = Parallel(n_jobs, verbose=0)
            my_func = delayed(func)
        except:
            n_jobs = 1

    # for a single job, or if we don't have joblib, don't use joblib
    if n_jobs == 1:
        n_jobs = 1
        my_func = func
        parallel = list
        return parallel, my_func, n_jobs

    return parallel, my_func, n_jobs


def _check_n_jobs(n_jobs):
    """Check n_jobs in particular for negative values

    Parameters
    ----------
    n_jobs : int
        The number of jobs.

    Returns
    -------
    n_jobs : int
        The checked number of jobs. Always positive.
    """
    if not isinstance(n_jobs, int):
        raise TypeError('n_jobs must be an integer')
    if n_jobs <= 0:
        import multiprocessing
        n_cores = multiprocessing.cpu_count()
        n_jobs = max(min(n_cores + n_jobs + 1, n_cores), 1)
    return n_jobs
