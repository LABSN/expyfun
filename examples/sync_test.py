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
                          stim_fs=24414, participant='s', session='0',
                          output_dir=None) as ec:
    ec.load_buffer(np.r_[0.1, np.zeros(2000)])
    black = [1, 1, 1]
    white = [-1, -1, -1]
    while True:
        ec.window.setColor(white)
        ec.flip_and_play()
        #ec.flip()                  # expyfun
        #ec._ac.play()
        ec.wait_one_press(0.5 + rng.rand(1) * 0.0167)
        ec.stop()
        ec.window.setColor(black)
        ec.flip()                  # expyfun
        ec.wait_one_press(rng.rand(1) * 5 * 0.0167)
