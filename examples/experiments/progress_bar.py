#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
================
ProgressBar demo
================

This example shows how to display progress between trials using
:class:`expyfun.visual.ProgressBar`.
"""
from expyfun import ExperimentController
from expyfun.visual import ProgressBar
import numpy as np

n_trials = 6

with ExperimentController('name', version='dev', window_size=[500, 500],
                          full_screen=False, session='foo',
                          participant='foo') as ec:

    # initialize the progress bar
    pb = ProgressBar(ec, [0, -.1, 1.5, .1], units='norm')

    ec.screen_prompt('Press the number shown on the screen. Start by pressing'
                     ' 1.', font_size=12, live_keys=[1])

    for n in np.arange(n_trials) + 1:
        # subject does some task
        number = np.random.randint(1, 5)
        ec.screen_text(str(number), wrap=False)
        ec.flip()
        ec.wait_one_press(live_keys=[number])
        ec.flip()
        ec.wait_secs(0.5)
        # only show progress bar every other trial
        if n % 2 == 0:
            # calculate percent done and update the bar object
            percent = n * 100 / n_trials
            pb.update_bar(percent)
            # display the progress bar with some text
            ec.screen_text('You\'ve completed {} %. Press any key to proceed.'
                           ''.format(percent), [0, .1], wrap=False,
                           font_size=12)
            pb.draw()
            ec.flip()
            # subject uses any key press to proceed
            ec.wait_one_press()
    ec.screen_text('This example is complete.')
    ec.flip()
    ec.wait_secs(1)
