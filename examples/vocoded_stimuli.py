# -*- coding: utf-8 -*-
"""
========================
Generate vocoded stimuli
========================

This shows how to make simple vocoded stimuli.

@author: larsoner
"""

import numpy as np

from expyfun.stimuli import vocode_ci, play_sound, window_edges, read_wav
from expyfun import fetch_data_file

data, fs = read_wav(fetch_data_file('audio/dream.wav'))
data = window_edges(data[0], fs)
t = np.arange(data.size) / float(fs)
data_ci = vocode_ci(data, fs, mode='noise', order=4, verbose=True)

# Uncomment this to play the original, too:
#snd = play_sound(data, fs, norm=False, wait=False)
snd = play_sound(data_ci, fs, norm=False, wait=False)

import matplotlib.pyplot as mpl
mpl.ion()
ax1 = mpl.subplot(2, 1, 1)
ax1.plot(t, data)
ax1.set_title('Original')
ax2 = mpl.subplot(2, 1, 2, sharex=ax1, sharey=ax1)
ax2.plot(t, data_ci)
ax2.set_title('Vocoded')
ax2.set_xlabel('Time (sec)')
