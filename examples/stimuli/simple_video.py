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

ec_args = dict(exp_name='simple video example', window_size=(720, 480),
               full_screen=False, participant='foo', session='foo',
               version='dev', enable_video=True, output_dir=None)

with ExperimentController(**ec_args) as ec:
    ec.load_video(movie_path)
    ec.video.set_scale('fit')
    t_zero = ec.video.play()
    while not ec.video.finished:
        if ec.video.playing:
            fliptime = ec.flip()
    ec.delete_video()
    ec.flip()
    ec.screen_prompt('video over', max_wait=1.)
