"""
=============
A-V sync test
=============
This example tests synchronization between the screen and the audio playback.
NOTE: On Linux (w/NVIDIA), XFCE has been observed to give consistent timings,
whereas Compiz WMs did not (doubled timings).
"""
# Author: Dan McCloy <drmccloy@uw.edu>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

from expyfun import ExperimentController
from expyfun.visual import Circle
import expyfun.analyze as ea

print(__doc__)


# Fullscreen MUST be used to guarantee flip accuracy!
with ExperimentController('SyncTest', full_screen=True, noise_db=-np.inf,
                          participant='s', session='0', output_dir=None,
                          suppress_resamp=True, check_rms=None,
                          version='dev') as ec:
    ec.load_buffer(np.r_[0.1, np.zeros(99)])  # RMS == 0.01
    pressed = None
    screenshot = None
    # Make a circle so that the photodiode can be centered on the screen
    circle = Circle(ec, 1, units='deg', fill_color='k', line_color='w')
    while pressed != '8':  # enable a clean quit if required
        ec.set_background_color('white')
        t1 = ec.start_stimulus(start_of_trial=False)  # skip checks
        ec.set_background_color('black')
        t2 = ec.flip()
        diff = round(1000 * (t2 - t1), 2)
        ec.screen_text('IFI (ms): {}'.format(diff), wrap=True)
        circle.draw()
        screenshot = ec.screenshot() if screenshot is None else screenshot
        ec.flip()
        pressed = ec.wait_one_press(0.5)[0]
        ec.stop()

plt.ion()
ea.plot_screen(screenshot)
