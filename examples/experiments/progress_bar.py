#!/usr/bin/env python2
"""
================
ProgressBar demo
================

This example shows how to display progress between trials using
:class:`expyfun.visual.ProgressBar`.
"""

import numpy as np

import expyfun.analyze as ea
from expyfun import ExperimentController, building_doc
from expyfun.visual import ProgressBar

n_trials = 6
max_wait = 0.1 if building_doc else np.inf
wait_dur = 0.1 if building_doc else 0.5

with ExperimentController(
    "name",
    version="dev",
    window_size=[800, 600],
    full_screen=False,
    session="foo",
    participant="foo",
) as ec:
    # initialize the progress bar
    pb = ProgressBar(ec, [0, -0.1, 1.5, 0.1], units="norm")

    ec.screen_prompt(
        "Press the number shown on the screen. Start by pressing 1.",
        font_size=16,
        live_keys=[1],
        max_wait=max_wait,
    )

    for n in np.arange(n_trials) + 1:
        # subject does some task
        number = np.random.randint(1, 5)
        ec.screen_text(str(number), wrap=False)
        ec.flip()
        ec.wait_one_press(live_keys=[number], max_wait=max_wait)
        ec.flip()
        ec.wait_secs(wait_dur)
        # only show progress bar every other trial
        if n % 2 == 0:
            # calculate percent done and update the bar object
            percent = int(n * 100 / n_trials)
            pb.update_bar(percent)
            # display the progress bar with some text
            ec.screen_text(
                f"You've completed {percent} %. Press any key to proceed.",
                [0, 0.1],
                wrap=False,
                font_size=16,
            )
            pb.draw()
            if n == 4:
                screenshot = ec.screenshot()
            ec.flip()
            # subject uses any key press to proceed
            ec.wait_one_press(max_wait=max_wait)
    ec.screen_text("This example is complete.")
    ec.flip()
    ec.wait_secs(1)

ea.plot_screen(screenshot)
