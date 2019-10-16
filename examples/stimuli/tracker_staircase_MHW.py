# -*- coding: utf-8 -*-
"""
=================================================
Do an adaptive track staircase with MHW procedure
=================================================

This shows how to make and use an adaptive track with the modified
Hughson-Westlake (MHW) procedure using
:class:`expyfun.stimuli.TrackerMHW`.

"""

import numpy as np

from expyfun.stimuli import TrackerMHW
from expyfun.analyze import sigmoid


# Make a callback function that prints to the console, rather than log file
def callback(event_type, value=None, timestamp=None):
    print((str(event_type) + ':').ljust(40) + str(value))


# Define parameters for modeled human subject (sigmoid probability)
true_thresh = 115
slope = 0.8
chance = 0.08  # if you don't hear it, you don't respond

# Make a tracker that uses the weighted up-down procedure to find 75%
tr = TrackerMHW(callback, 0, 120, base_step=5, start_value=80)

# Initialize human state
rng = np.random.RandomState(1)

# Do the task until the tracker stops
while not tr.stopped:
    tr.respond(rng.rand() < sigmoid(tr.x_current - true_thresh,
                                    lower=chance, slope=slope))

# Plot the results
fig, ax, lines = tr.plot()
lines += tr.plot_thresh()

ax.set_title('Adaptive track of model human (true threshold is {})'
             .format(true_thresh))
