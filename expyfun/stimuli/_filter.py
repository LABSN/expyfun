# -*- coding: utf-8 -*-
"""Stimulus resampling functions
"""


def _resample_error(*args, **kwargs):
    """mne-python is required to use the resample function
    """
    raise ImportError('mne-python is required to use the resample function')

try:
    from mne.filter import resample
except ImportError:
    resample = _resample_error
