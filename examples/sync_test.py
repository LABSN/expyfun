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

print(__doc__)

import numpy as np
from expyfun import ExperimentController
import expyfun.analyze as ea

ac = 'pyglet'
stim_fs = 44100
#ac = dict(TYPE='tdt', TDT_MODEL='RM1')
#stim_fs = 24414

tc = 'dummy'
#tc = 'parallel'
#tc = 'tdt'

# Fullscreen MUST be used to guarantee flip accuracy!
with ExperimentController('SyncTest', screen_num=0, full_screen=True,
                          stim_db=90, noise_db=-np.inf, stim_fs=stim_fs,
                          participant='s', session='0', audio_controller=ac,
                          trigger_controller=tc, output_dir=None,
                          suppress_resamp=True, check_rms=None) as ec:
    ec.load_buffer(np.r_[0.1, np.zeros(99)])  # RMS == 0.01
    pressed = None
    screenshot = None
    while pressed != '8':  # enable a clean quit if required
        ec.draw_background_color('white')
        t1 = ec.start_stimulus(start_of_trial=False)  # skip checks
        ec.draw_background_color('black')
        t2 = ec.flip()
        diff = round(1000 * (t2 - t1), 2)
        ec.screen_text('<center><b>IFI (ms):<br>{}</b></center>'.format(diff))
        if screenshot is None:
            screenshot = ec.screenshot()
        ec.flip()
        pressed = ec.wait_one_press(0.5)[0]
        ec.stop()
        ec.trial_ok()

import matplotlib.pyplot as plt
plt.ion()
ea.plot_screen(screenshot)
