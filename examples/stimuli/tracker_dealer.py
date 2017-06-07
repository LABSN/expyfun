# -*- coding: utf-8 -*-
"""
======================
Tracker Dealer Example
======================

This file shows how to interleave multiple Tracker objects using
:class:`expyfun.stimuli.TrackerDealer`.

In this case, a modeled human subject generates two curves (one for each trial
type: 1 & 2)

@author: maddycapp27
"""

import numpy as np
from expyfun.stimuli import TrackerUD, TrackerDealer
from expyfun.analyze import sigmoid
import matplotlib.pyplot as plt

# define parameters of modeled subject (using sigmoid probability)
true_thresh = [30, 40]
slope = 0.1
chance = 0.5

# define parameters for each tracker (assuming each tracker uses same rules)
up = 1
down = 1
step_size_up = [9, 3]
step_size_down = [3, 1]
stop_criterion = 30
stop_rule = 'reversals'
start_value = 45
change_criteria = [0, 5]
change_rule = 'reversals'
x_min = 0
x_max = 90

# parameters for the tracker dealer
max_lag = 2
rand = None


# callback function that prints to console
def callback(event_type, value=None, timestamp=None):
    print((str(event_type) + ':').ljust(40) + str(value))


# initialize two tracker objects--one for each trial type
tr_UD = [TrackerUD(callback, up, down, step_size_up, step_size_down,
                   stop_criterion, stop_rule, start_value,
                   change_criteria, change_rule, x_min, x_max) for i in [0, 1]]

# initialize TrackerDealer object
tr = TrackerDealer(tr_UD, max_lag, rand)

# Initialize human state
rng = np.random.RandomState(1)

while not tr.stopped:
    # Get information of which trial type is next and what the level is at
    # that time from TrackerDealer
    ss, level = tr.get_trial()
    ss = sum(ss)
    tr_UD[ss].respond(rng.rand() < sigmoid(level - true_thresh[ss],
                                           lower=chance, slope=slope))

# Plot the results
axes = [plt.subplot(211), plt.subplot(212)]
for i in [0, 1]:
    fig, ax, lines = tr[i].plot(ax=axes[i])
    lines += tr[i].plot_thresh(4, ax=ax)

    lines[0].set_label('Trials')
    lines[1].set_label('Reversals')
    lines[2].set_label('Estimated threshold')

    ax.legend(loc='best')
    ax.set_title('Adaptive track of model human trial type {} (true threshold '
                 'is {})'.format(i + 1, true_thresh[i]))
    plt.tight_layout()
