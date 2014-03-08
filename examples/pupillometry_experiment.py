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
from expyfun.analyze import sigmoid
from expyfun.codeblocks import (find_pupil_dynamic_range,
                                find_pupil_impulse_response)


with ExperimentController('testExp', full_screen=True, participant='foo',
                          session='001', output_dir=None) as ec:
    el = EyelinkController(ec)
    ec.screen_prompt('Welcome to the experiment!<br><br>First, we will '
                     'perform a screen calibration.<br><br>Press a button '
                     'to continue.')
    fname = el.calibrate()  # by default this starts recording EyeLink data

    ec.screen_prompt('Excellent! Now, we will determine the dynamic '
                     'range of your pupil.<br><br>Press a button to continue.')
    lin_reg, lev, resp, params = find_pupil_dynamic_range(ec, el, 0.01,
                                                          fname)
    # XXX SHORT

    ec.screen_prompt('Now, we will determine the impulse response '
                     'of your pupil.<br><br>Press a button to continue.')
    prf, screen_fs = find_pupil_impulse_response(ec, el, lin_reg,
                                                 max_dur=3., n_repeats=1)
    # XXX MORE REPEATS

import matplotlib.pyplot as plt
plt.ion()
uni_lev = np.unique(lev)
# Grayscale responses
plt.subplot(2, 1, 1, xlabel='Screen level', ylabel='Pupil dilation (AU)')
plt.plot(lev, resp, linestyle='none', marker='o', color='r')
plt.plot(uni_lev, sigmoid(uni_lev, *params), color='k')
# PRF
plt.subplot(2, 1, 2, xlabel='Time (s)', ylabel='Pupil response function (AU)')
t = np.arange(len(prf)) / screen_fs
plt.plot(t, prf, color='k')
plt.xlim(t[[0, -1]])
plt.tight_layout()
