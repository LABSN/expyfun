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

ac = 'psychopy'
#ac = dict(TYPE='tdt', TDT_MODEL='RM1')

tc = 'dummy'
#tc = 'parallel'
#tc = 'tdt'

# Fullscreen MUST be used to guarantee flip accuracy!
with ExperimentController('SyncTest', screen_num=0, full_screen=True,
                          stim_db=90, noise_db=-np.inf, stim_fs=24414,
                          participant='s', session='0', audio_controller=ac,
                          trigger_controller=tc, output_dir=None,
                          suppress_resamp=True) as ec:
    ec.load_buffer(np.r_[0.1, np.zeros(2000)])
    while True:
        ec.draw_background_color('white')
        t1 = ec.flip_and_play()
        ec.draw_background_color('black')
        t2 = ec.flip()
        diff = round(1000 * (t2 - t1), 2)
        ec.screen_prompt('\n\n\nIFI (ms):\n{}'.format(diff),
                         0, clear_after=False)
        ec.wait_one_press(0.5)
        ec.stop()
