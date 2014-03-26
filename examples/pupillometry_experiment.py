"""
==========================================
Experiment using eye-tracking pupillometry
==========================================

Integration with Eyelink functionality makes programming experiments
using eye-tracking simpler.
"""
# Author: Eric Larson <larsoner@uw.edu>
#
# License: BSD (3-clause)

print(__doc__)

import numpy as np
from expyfun import ExperimentController, EyelinkController
from expyfun.codeblocks import (find_pupil_dynamic_range,
                                find_pupil_tone_impulse_response)


with ExperimentController('testExp', full_screen=True, participant='foo',
                          session='001', output_dir=None) as ec:
    el = EyelinkController(ec)
    fname = el.calibrate()  # by default this starts recording EyeLink data
    bgcolor, lev, resp = find_pupil_dynamic_range(ec, el)
    prf, p_srf = find_pupil_tone_impulse_response(ec, el, bgcolor)

import matplotlib.pyplot as plt
plt.ion()
uni_lev = np.unique(lev)
# Grayscale responses
plt.subplot(2, 1, 1, xlabel='Screen level', ylabel='Pupil dilation (AU)')
plt.plot(lev, resp, linestyle='none', marker='o', color='r')
# PRF
plt.subplot(2, 1, 2, xlabel='Time (s)', ylabel='Pupil response function (AU)')
plt.plot(p_srf, prf, color='k')
plt.xlim(p_srf[[0, -1]])
plt.tight_layout()