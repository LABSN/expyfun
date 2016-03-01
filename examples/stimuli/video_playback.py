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
    ec.load_video(movie_path)
    ec.video.play()
    #ec.video.seek(1.)

    while ec.video.playing:
        next_frame = ec.video.last_frame + ec.video.dt
        ec.flip(when=next_frame)
    ec.screen_prompt('video over!', max_wait=1.)
