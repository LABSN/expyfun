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
    while True:
        ec.window.setColor([1, 1, 1])
        #ec.flip_and_play()
        #ec.flip()                  # expyfun
        #ec.window.flip()           # psychopy
        ec.window.winHandle.flip()  # pyglet
        ec._ac.play()
        ec.wait_one_press(rng.rand(1) * 0.0167)
        ec.stop()
        ec.window.setColor([-1, -1, -1])
        #ec.flip()                  # expyfun
        #ec.window.flip()           # psychopy
        ec.window.winHandle.flip()  # pyglet
        ec.wait_one_press(rng.rand(1) * 0.0167)
