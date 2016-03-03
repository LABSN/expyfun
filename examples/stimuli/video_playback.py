# -*- coding: utf-8 -*-
"""
======================
Play sample video file
======================

This shows how to play a video file in expyfun.

@author: drmccloy
"""

from expyfun import ExperimentController, fetch_data_file

print(__doc__)


movie_path = fetch_data_file('video/example-video.mp4')

ec_args = dict(exp_name='movietest', window_size=(720, 480),
               full_screen=False, participant='foo', session='foo',
               version='dev', enable_video=True)

with ExperimentController(**ec_args) as ec:
    all_presses = list()
    ec.load_video(movie_path)
    ec.screen_prompt('press 1 during video to toggle pause.', max_wait=1.)
    ec.video.play()
    t_zero = ec.get_time()

    while ec.video.time < ec.video.duration:
        ec.listen_presses()
        if ec.video.playing:
            t = ec.get_time()
            fliptime = (ec.video.next_timestamp if ec.video.next_timestamp > t
                        else None)
            ec.flip(when=fliptime)
        ec.wait_secs(0.0005)  # enough time for a press to register
        presses = ec.get_presses(live_keys=[1], relative_to=t_zero)
        # presses won't be caught during this part of the while loop!
        if len(presses):
            all_presses.extend(presses)
            if ec.video.playing:
                ec.screen_text('pause!')
                ec.flip()
                ec.video.pause()
            else:
                ec.screen_text('play!')
                ec.video.play()
    ec.unload_video()
    preamble = 'press times:' if len(all_presses) else 'no presses'
    msg = ', '.join(['{0:.3f}'.format(x[1]) for x in all_presses])
    ec.flip()
    ec.screen_prompt('\n'.join([preamble, msg]), max_wait=1.)
