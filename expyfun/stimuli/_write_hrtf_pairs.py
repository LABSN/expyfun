#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 19:20:21 2017

@author: Maddy
"""

import numpy as np
from expyfun.io import read_hdf5, write_hdf5
from expyfun._utils import fetch_data_file

source = 'cipic'  # interpolation only works with cipic library
fs = 44100  # work with each sampling frequency to manually correct unwrapping

fname = fetch_data_file('hrtf/{0}_{1}.hdf5'.format(source, fs))
data = read_hdf5(fname)
angles = data['angles']
brir = data['brir']

hrtf_phase = []
hrtf_amp = []
pairs = []

for read_angle in np.arange(1, 90, 5):  # cover all pairs
    idx = np.where(angles < read_angle)[0]
    # angles with known hrtfs to interpolate between
    known_angles = np.array([angles[idx[-1]], angles[idx[-1] + 1]])
    pairs.append(known_angles)

    # get known brirs for those angles, convert to frequency domain,
    # separate out magnitude and phase
    brirs = [brir[idx[-1]].copy(), brir[idx[-1] + 1].copy()]
    hrtfs = np.fft.fft(brirs)
    hrtf_phase.append(np.unwrap(np.angle(hrtfs)))
    hrtf_amp.append(np.abs(hrtfs))

    ######################################
    # MAKE MANUAL UNWRAPPING CORRECTIONS #
    ######################################
    if read_angle < 90 and read_angle > 85:

        hrtf_phase[-1][1][1, 41:] += -2 * np.pi

    elif read_angle < 85 and read_angle > 80:

        hrtf_phase[-1][0][1, 91:] += -2 * np.pi
        hrtf_phase[-1][1][0, 35:] += -2 * np.pi
        hrtf_phase[-1][0][0, 25:] += -2 * np.pi

    elif read_angle < 70 and read_angle > 65:

        hrtf_phase[-1][0][1, 49:] += -2 * np.pi
        hrtf_phase[-1][0][0, 92:] += -2 * np.pi
        hrtf_phase[-1][1][0, 42:] += -2 * np.pi

    elif read_angle < 65 and read_angle > 60:

        hrtf_phase[-1][0][0, 42:] += -2 * np.pi

    elif read_angle < 35 and read_angle > 30:

        hrtf_phase[-1][1][1, 73:] += -2 * np.pi

#########################################
# SAVE AS UNWRAPPED PHASE AND AMPLITUDE #
#########################################
hrtf_dict = {'hrtf_phase': hrtf_phase, 'hrtf_amp': hrtf_amp, 'fs': fs,
             'angles': angles, 'pairs': pairs}
write_hdf5('/Users/Maddy/.expyfun/data/hrtf/pair_cipic_{0}.hdf5'.format(fs),
           hrtf_dict, overwrite=True)
# %%
fs = 24414

fname = fetch_data_file('hrtf/{0}_{1}.hdf5'.format(source, fs))
data = read_hdf5(fname)
angles = data['angles']
brir = data['brir']

hrtf_phase = []
hrtf_amp = []
pairs = []

for read_angle in np.arange(1, 90, 5):  # cover all pairs
    idx = np.where(angles < read_angle)[0]
    # angles are
    known_angles = np.array([angles[idx[-1]], angles[idx[-1] + 1]])
    pairs.append(known_angles)

    # get known brirs
    brirs = [brir[idx[-1]].copy(), brir[idx[-1] + 1].copy()]
    hrtfs = np.fft.fft(brirs)
    hrtf_phase.append(np.unwrap(np.angle(hrtfs)))
    hrtf_amp.append(np.abs(hrtfs))

    if read_angle < 90 and read_angle > 85:

        hrtf_phase[-1][0][0, 51:] += -2 * np.pi
        hrtf_phase[-1][1][0, 50:] += -2 * np.pi

        hrtf_phase[-1][1][1, 41:] += -2 * np.pi
        hrtf_phase[-1][1][1, 51:] += -2 * np.pi

    elif read_angle < 85 and read_angle > 80:

        hrtf_phase[-1][1][0, 35:] += -2 * np.pi
        hrtf_phase[-1][0][0, 24:] += -2 * np.pi
        hrtf_phase[-1][1][0, 51:] += -2 * np.pi

    elif read_angle < 70 and read_angle > 65:

        hrtf_phase[-1][1][0, 43:] += -2 * np.pi

    elif read_angle < 65 and read_angle > 60:
        
        hrtf_phase[-1][0][0, 41:] += -2 * np.pi

#########################################
# SAVE AS UNWRAPPED PHASE AND AMPLITUDE #
#########################################
hrtf_dict = {'hrtf_phase': hrtf_phase, 'hrtf_amp': hrtf_amp, 'fs': fs,
             'angles': angles, 'pairs': pairs}
write_hdf5('/Users/Maddy/.expyfun/data/hrtf/pair_cipic_{0}.hdf5'.format(fs),
           hrtf_dict, overwrite=True)
