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

from expyfun import ExperimentController, EyelinkController
from expyfun.codeblocks import (find_pupil_dynamic_range,
                                find_pupil_impulse_response)


with ExperimentController('testExp', full_screen=True, participant='foo',
                          session='001', output_dir=None) as ec:
    el = EyelinkController(ec)
    #ec.screen_prompt('Welcome to the experiment!<br><br>First, we will '
    #                 'perform a screen calibration.<br><br>Press a button '
    #                 'to continue.')
    #el.calibrate()  # by default this starts recording EyeLink data

    ec.screen_prompt('Excellent! Now, we will determine the dynamic '
                     'range of your pupil.<br><br>Press a button to continue.')
    lin_reg = find_pupil_dynamic_range(ec, el, 0.01)  # XXX TOO SHORT

    ec.screen_prompt('Now, we will determine the impulse response '
                     'of your pupil.<br><br>Press a button to continue.')
    impulse_response = find_pupil_impulse_response(ec, el, lin_reg)

import matplotlib.pyplot as plt
plt.ion()
