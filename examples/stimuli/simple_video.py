# -*- coding: utf-8 -*-
"""
=========================
Video playing made simple
=========================

This shows how to play a video file in expyfun. It requires that FFmpeg
(for Pyglet >= 1.4) or AVBin (otherwise) has already been installed.

@author: drmccloy
"""

from expyfun import (ExperimentController, fetch_data_file, analyze as ea,
                     building_doc)

print(__doc__)

movie_path = fetch_data_file('video/example-video.mp4')

ec_args = dict(exp_name='simple video example', window_size=(720, 480),
               full_screen=False, participant='foo', session='foo',
               version='dev', output_dir=None)
screenshot = None

with ExperimentController(**ec_args) as ec:
    ec.load_video(movie_path)
    ec.video.set_scale('fit')
    t_zero = ec.video.play()
    while not ec.video.finished:
        if ec.video.playing:
            fliptime = ec.flip()
        if screenshot is None:
            screenshot = ec.screenshot()
        if building_doc:
            break
    ec.delete_video()
    ec.flip()
    ec.screen_prompt('video over', max_wait=1.)

ea.plot_screen(screenshot)
