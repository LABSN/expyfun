# -*- coding: utf-8 -*-
"""
======================
Play sample video file
======================

This shows how to play a video file in expyfun.

@author: drmccloy
"""

import numpy as np
from expyfun import ExperimentController, fetch_data_file

print(__doc__)


movie_path = fetch_data_file('video/example-video.mp4')

ec_args = dict(exp_name='movietest', window_size=(720, 480),
               full_screen=False, participant='foo', session='foo',
               version='dev', enable_video=True)

with ExperimentController(**ec_args) as ec:
    screen_period = 1. / ec.estimate_screen_fs()
    all_presses = list()
    ec.load_video(movie_path)
    ec.screen_prompt('press 1 during video to toggle pause.', max_wait=1.)
    ec.listen_presses()
    t_zero = ec.video.play()
    while not ec.video.finished:
        if ec.video.playing:
            fliptime = ec.flip()
        else:  # to catch presses reliably, need delay between loop executions
            ec.wait_secs(screen_period / 5)
        presses = ec.get_presses(live_keys=[1], relative_to=t_zero)
        ec.listen_presses()
        if ec.video.playing and 3 < ec.video.time < 4.5:
            ec.video.set_scale(ec.video.scale * 0.99)
        if ec.video.playing and ec.video.time > 4.5:
            ec.video.set_pos(ec.video.position + np.array((0.01, 0)))
        if len(presses):
            all_presses.extend(presses)
            if ec.video.playing:
                ec.screen_text('pause!')
                ec.flip()
                ec.video.pause()
            else:
                ec.screen_text('play!')
                ec.video.play()
    ec.delete_video()
    preamble = 'press times:' if len(all_presses) else 'no presses'
    msg = ', '.join(['{0:.3f}'.format(x[1]) for x in all_presses])
    ec.flip()
    ec.screen_prompt('\n'.join([preamble, msg]), max_wait=1.)
