# -*- coding: utf-8 -*-
"""Stimulus resampling functions
"""

try:
    from mne.filter import resample as _resample
except ImportError:
    _resample = None


def resample(*args, **kwargs):
    _resample.__doc__
    return _resample(*args, **kwargs)

# Get the current docstring from mne, rather than a potentially stale one
if _resample is not None:
    resample.__doc__ = _resample.__doc__
