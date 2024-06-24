"""
Sound stimulus design
=====================

Experiment sound stimulus functions.
"""

# -*- coding: utf-8 -*-
# Copyright (c) 2014, LABSN.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

from ._hrtf import convolve_hrtf
from ._mls import compute_mls_impulse_response, repeated_mls
from ._stimuli import rms, play_sound, window_edges, add_pad
from ._vocoder import vocode, get_band_freqs, get_bands, get_env, get_carriers
from ._tracker import TrackerUD, TrackerBinom, TrackerDealer, TrackerMHW
from .._tdt_controller import get_tdt_rates
from ._texture import texture_ERB
from ._crm import (
    crm_sentence,
    crm_response_menu,
    crm_prepare_corpus,
    crm_info,
    CRMPreload,
)

# for backward compat (not great to do this...)
from ..io import read_wav, write_wav
