# -*- coding: utf-8 -*-
"""
======================================
Adaptive tracking from above and below
======================================

This file shows how to interleave multiple Tracker objects using
:class:`expyfun.stimuli.TrackerDealer` to simultaneously approach a threshold
from both above and below.

@author: maddycapp27
"""

import numpy as np
from expyfun.stimuli import TrackerUD, TrackerDealer
from expyfun.analyze import sigmoid
import matplotlib.pyplot as plt

# define parameters of modeled subject (using sigmoid probability)
true_thresh = 30  # true thresholds for trial types 1 and 2
slope = 0.1
chance = 0.5

##############################################################################
# Defining Tracker Parameters
# ---------------------------
# In this example, the tracker parameters are the same for each instance of
# the up-down adaptive tracker except for the start value. Each start value in
# the list will be given to a different tracker. The other parameters are
# defined such that the step sizes vary for both up v. down (the up step size
# is larger by a factor of 3) and based on the number of reversals (the first
# element in each list is the step size until the number of reversals dictated
# by the second element in change_criteria have occurred (i.e. the up step size
# will be 9 until 5 reversals have occurred, then the up step size will be 3.))
up = 1
down = 1
step_size_up = [9, 3]
step_size_down = [3, 1]
stop_reversals = 30
stop_trials = np.inf
start_value = [15, 45]
change_indices = [5]
change_rule = 'reversals'
x_min = 0
x_max = 90


# callback function that prints to console
def callback(event_type, value=None, timestamp=None):
    print((str(event_type) + ':').ljust(40) + str(value))


# parameters for the tracker dealer
max_lag = 2
pace_rule = 'reversals'
rng_dealer = np.random.RandomState(4)  # random seed for selecting trial type

##############################################################################
# Initializing and Running Trackers
# ---------------------------------
# The two trackers in this example use all of the same parameters except for
# the start value and then are passed into the dealer. After the dealer is
# created, the type of trial with the start value above or below the true
# threshold (returned as an index) and trial level for that trial can be
# acquired.

# initialize two tracker objects--one for each start value
tr_ud = [TrackerUD(callback, up, down, step_size_up, step_size_down,
                   stop_reversals, stop_trials, sv, change_indices,
                   change_rule, x_min, x_max) for sv in start_value]

# initialize TrackerDealer object
td = TrackerDealer(callback, tr_ud, max_lag, pace_rule, rng_dealer)

# Initialize human state
rng_human = np.random.RandomState(1)  # random seed for modeled subject

for _, level in td:
    # Get information of which trial type is next and what the level is at
    # that time from TrackerDealer
    td.respond(rng_human.rand() < sigmoid(level - true_thresh,
                                          lower=chance, slope=slope))

##############################################################################
# Plotting the Results
# ---------------------------
axes = plt.subplots(2, 1)[1]
for i in [0, 1]:
    fig, ax, lines = td.trackers.ravel()[i].plot(ax=axes[i], n_skip=4)

    ax.legend(loc='best')
    ax.set_title('Adaptive track with start value {} (true threshold '
                 'is {})'.format(start_value[i], true_thresh))
    fig.tight_layout()
