# -*- coding: utf-8 -*-
"""Stimulus resampling functions
"""

from distutils.version import LooseVersion

import mne
if LooseVersion(mne.__version__) < LooseVersion('0.8'):
    raise ImportError('mne-python too old')
from mne.filter import resample  # NOQA
