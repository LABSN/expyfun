# -*- coding: utf-8 -*-
"""
===========================
Do a couple adaptive tracks
===========================

This shows how to make and use an adaptive track using AdaptiveTracker.

@author: rkmaddox
"""

import numpy as np

from expyfun.stimuli import TrackerUD
from expyfun.analyze import sigmoid

print(__doc__)


# Make a callback function that prints to the console, rather than log file
def callback(event_type, value=None, timestamp=None):
    print(event_type, value, timestamp)

# Define parameters for modeled human subject (sigmoid probability)
true_thresh = 35
slope = 0.1
chance = 0.5

# Make a tracker that uses the weighted up-down procedure to find 75%
tr = TrackerUD(callback, 1, 1, [9, 3], [3, 1], 30, 'reversals', 60, [0, 4])

# Initialize human state
rand = np.random.RandomState(1)

# Do the task until the tracker stops
while not tr.stopped:
    tr.respond(rand.rand() < sigmoid(tr.x_current - true_thresh,
                                     lower=0.5, slope=0.1))

# Plot the results
fig, ax, lines = tr.plot()
lines += tr.plot_thresh(4, ax=ax)

lines[0].set_label('Trials')
lines[1].set_label('Reversals')
lines[2].set_label('Estimated threshold')

ax.legend()
ax.set_title('Adaptive track of model human (true threshold is {})'
             .format(true_thresh))
