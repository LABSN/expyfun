"""
=============
A-V sync test
=============

This example tests synchronization between the screen and the audio playback.
"""
# Author: Dan McCloy <drmccloy@uw.edu>
#
# License: BSD (3-clause)

print __doc__

import numpy as np
from expyfun import ExperimentController

rng = np.random.RandomState(0)

with ExperimentController('SyncTest', screen_num=0, window_size=[300, 300],
                          full_screen=False, stim_db=70, noise_db=-np.inf,
                          stim_fs=44100, participant='s', session='0',
                          output_dir=None) as ec:
    ec.load_buffer(np.r_[0.1, np.zeros(2000)])
    white = [1, 1, 1]
    black = [-1, -1, -1]
    while True:
        ec.draw_background_color(white)
        t1 = ec.flip_and_play()
        ec.draw_background_color(black)
        t2 = ec.flip()                  # expyfun
        print 1. / (t2 - t1)
        ec.wait_one_press(0.5)
