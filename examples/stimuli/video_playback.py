# -*- coding: utf-8 -*-
"""
===============================================================================
Script ''
===============================================================================

This script does XXX.
"""
# @author: drmccloy
# Created on Tue Feb 23 13:57:53 2016
# License: BSD (3-clause)

from __future__ import print_function
from expyfun import ExperimentController as EC

movie_path = '/home/drmccloy/Videos/trimmed/shaun-the-sheep-01.m4v'

ec_args = dict(exp_name='movietest', window_size=(720, 480),
               full_screen=False, participant='foo', session='foo',
               version='dev', enable_video=True)
ec = EC(**ec_args)
ec.load_video(movie_path)
ec.video.play()
#ec.video.seek(15.)

while ec.video.playing:
    next_frame = ec.video.last_frame + ec.video.dt
    _ = ec.flip(when=next_frame)
