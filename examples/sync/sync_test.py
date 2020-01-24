"""
=============
A-V sync test
=============

This example tests synchronization between the screen and the audio playback.

.. note:: On Linux (w/NVIDIA), XFCE has been observed to give consistent
          timings, whereas Compiz WMs did not (doubled timings).
"""
# Author: Dan McCloy <drmccloy@uw.edu>
#
# License: BSD (3-clause)

import numpy as np

from expyfun import ExperimentController, building_doc
from expyfun.visual import Circle, Rectangle
import expyfun.analyze as ea

print(__doc__)


# Fullscreen MUST be used to guarantee flip accuracy!
n_channels = 2
click_idx = [0]
with ExperimentController('SyncTest', full_screen=True, noise_db=-np.inf,
                          participant='s', session='0', output_dir=None,
                          suppress_resamp=True, check_rms=None,
                          n_channels=n_channels, version='dev') as ec:
    click = np.r_[0.1, np.zeros(99)]  # RMS = 0.01
    data = np.zeros((n_channels, len(click)))
    data[click_idx] = click
    ec.load_buffer(data)
    pressed = None
    screenshot = None
    # Make a circle so that the photodiode can be centered on the screen
    circle = Circle(ec, 1, units='deg', fill_color='k', line_color='w')
    # Make a rectangle that is the standard credit card size (~3 3/8" x 2 1/8")
    rect = Rectangle(ec, [0, 0, 8.56, 5.398], 'cm', None, '#AA3377')
    while pressed != '8':  # enable a clean quit if required
        ec.set_background_color('white')
        t1 = ec.start_stimulus(start_of_trial=False)  # skip checks
        ec.set_background_color('black')
        t2 = ec.flip()
        diff = round(1000 * (t2 - t1), 2)
        ec.screen_text('IFI (ms): {}'.format(diff), wrap=True)
        circle.draw()
        rect.draw()
        screenshot = ec.screenshot() if screenshot is None else screenshot
        ec.flip()
        ec.stamp_triggers([2, 4, 8])
        ec.refocus()
        pressed = ec.wait_one_press(0.5)[0] if not building_doc else '8'
        ec.stop()

ea.plot_screen(screenshot)
