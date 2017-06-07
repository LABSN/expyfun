# -*- coding: utf-8 -*-
"""
======================
Tracker Dealer Example
======================

This file shows how to interleave multiple Tracker objects using
:class:`expyfun.stimuli.TrackerDealer`.

In this case, a modeled human subject generates two curves (one for each trial
type: 1 & 2).

@author: maddycapp27
"""

import numpy as np
from expyfun.stimuli import TrackerUD, TrackerDealer
from expyfun.analyze import sigmoid
import matplotlib.pyplot as plt

# define parameters of modeled subject (using sigmoid probability)
true_thresh = [30, 40]  # true thresholds for trial types 1 and 2
slope = 0.1
chance = 0.5

##############################################################################
# Defining Tracker Parameters
# ---------------------------
# In this example, the tracker parameters are exactly the same for each
# instance of the up-down adaptive tracker. These are defined such that the
# step sizes vary for both up v. down (the up step size is larger by a factor
# of 3) and based on the number of reversals (the first element in each
# list is the step size until the number of reversals dictated by the second
# element in change_criteria have occured (i.e. the up step size will be 9
# until 5 reversals have occured, then the up step size will be 3.))
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


# callback function that prints to console
def callback(event_type, value=None, timestamp=None):
    print((str(event_type) + ':').ljust(40) + str(value))


# parameters for the tracker dealer
max_lag = 2
rng_dealer = np.random.RandomState(3)  # random seed for selecting trial type

##############################################################################
# Initializing and Running Trackers
# ---------------------------------
# The two trackers in this example use all of the same parameters and then are
# passed into the dealer. After the dealer is created, the type of each trial
# (returned as an index of the array of individual trackers) and trial level
# for that trial can be acquired.

# initialize two tracker objects--one for each trial type
tr_UD = [TrackerUD(callback, up, down, step_size_up, step_size_down,
                   stop_criterion, stop_rule, start_value,
                   change_criteria, change_rule, x_min, x_max) for i in [0, 1]]

# initialize TrackerDealer object
tr = TrackerDealer(tr_UD, max_lag, rng_dealer)

# Initialize human state
rng_human = np.random.RandomState(1)  # random seed for modeled subject

while not tr.stopped:
    # Get information of which trial type is next and what the level is at
    # that time from TrackerDealer
    ss, level = tr.get_trial()
    ss = sum(ss)
    tr_UD[ss].respond(rng_human.rand() < sigmoid(level - true_thresh[ss],
                                                 lower=chance, slope=slope))

##############################################################################
# Plotting the Results
# ---------------------------
axes = plt.subplots(2, 1)[1]
for i in [0, 1]:
    fig, ax, lines = tr[i].plot(ax=axes[i])
    lines += tr[i].plot_thresh(4, ax=ax)

    lines[0].set_label('Trials')
    lines[1].set_label('Reversals')
    lines[2].set_label('Estimated threshold')

    ax.legend(loc='best')
    ax.set_title('Adaptive track of model human trial type {} (true threshold '
                 'is {})'.format(i + 1, true_thresh[i]))
    fig.tight_layout()
