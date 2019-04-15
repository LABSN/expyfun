# -*- coding: utf-8 -*-
"""
==========================================================================
Adaptive tracking for two trial types and tracker reconstruction from .tab
==========================================================================

This file shows how to interleave multiple Tracker objects using
:class:`expyfun.stimuli.TrackerDealer` as well as how to reconstruct the
dealer from the .tab file logged by experiment controller with
:func:`expyfun.io.reconstruct_dealer`,

In this case, a modeled human subject generates two curves (one for each trial
type: 1 & 2).

@author: maddycapp27
"""

import numpy as np
from expyfun import ExperimentController
from expyfun.stimuli import TrackerUD, TrackerDealer
from expyfun.analyze import sigmoid
from expyfun.io import reconstruct_dealer
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
# list is the step size until the number of reversals dictacted by the second
# element in change_criteria have occurred (i.e. the up step size will be 9
# until 5 reversals have occurred, then the up step size will be 3.))
up = 1
down = 1
step_size_up = [9, 3]
step_size_down = [3, 1]
stop_reversals = 30
stop_trials = np.inf
start_value = 45
change_indices = [5]
change_rule = 'reversals'
x_min = 0
x_max = 90

# parameters for the tracker dealer
max_lag = 2
pace_rule = 'reversals'
rng_dealer = np.random.RandomState(3)  # random seed to select trial type

###############################################################################
# Initializing and Running Trackers
# ---------------------------------
# The two trackers in this example use all of the same parameters and then are
# passed into the dealer. After the dealer is created, the type of each trial
# (returned as an index of the array of individual trackers) and trial level
# for that trial can be acquired. :class:`expyfun.ExperimentController` is used
# to generate log files with :class:`expyfun.stimuli.TrackerUD` and
# :class:`expyfun.stimuli.TrackerDealer` information.
std_args = ['test']  # experiment name
std_kwargs = dict(full_screen=False, window_size=(1, 1), participant='foo',
                  session='01', stim_db=0.0, noise_db=0.0, verbose=True,
                  version='dev')

with ExperimentController(*std_args, **std_kwargs) as ec:

    # initialize two tracker objects--one for each trial type
    tr_ud = [TrackerUD(ec, up, down, step_size_up, step_size_down,
                       stop_reversals, stop_trials, start_value,
                       change_indices, change_rule, x_min,
                       x_max) for _ in range(2)]

    # initialize TrackerDealer object
    td = TrackerDealer(ec, tr_ud, max_lag, pace_rule, rng_dealer)

    # Initialize human state
    rng_human = np.random.RandomState(1)  # random seed for modeled subject

    for ss, level in td:
        # Get information of which trial type is next and what the level is at
        # that time from TrackerDealer
        td.respond(rng_human.rand() < sigmoid(level - true_thresh[sum(ss)],
                                              lower=chance, slope=slope))

###############################################################################
# Reconstructing the TrackerDealer Object
# ---------------------------------------
# The TrackerDealer object has many built in analysis functions that are can
# only be access through the object itself (not the log files alone). By using
# :func:`expyfun.io.reconstruct_dealer`, the object can be recreated such that
# the analysis functions are accessible. Note that the function always returns
# a list of objects. Similar reconstructions of single trackers can be done
# with :func:`expyfun.io.reconstruct_tracker`.

td_tab = reconstruct_dealer(ec.data_fname)[0]

##############################################################################
# Plotting the Results
# ---------------------------
axes = plt.subplots(2, 1)[1]
for i in [0, 1]:
    fig, ax, lines = td_tab.trackers.ravel()[i].plot(ax=axes[i], n_skip=4)

    ax.legend(loc='best')
    ax.set_title('Adaptive track of model human trial type {} (true threshold '
                 'is {})'.format(i + 1, true_thresh[i]))
    fig.tight_layout()
