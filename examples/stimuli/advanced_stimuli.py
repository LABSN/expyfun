# -*- coding: utf-8 -*-
"""
=======================================
Generate more advanced auditory stimuli
=======================================

This shows the methods that we provide that facilitate generation
of more advanced stimuli.
"""

import numpy as np
import matplotlib.pyplot as plt

from expyfun import building_doc
from expyfun.stimuli import convolve_hrtf, play_sound, window_edges

fs = 24414
dur = 0.5
freq = 500.
# let's make a square wave
sig = np.sin(freq * 2 * np.pi * np.arange(dur * fs, dtype=float) / fs)
sig = ((sig > 0) - 0.5) / 5.  # make it reasonably quiet for play_sound
sig = window_edges(sig, fs)

play_sound(sig, fs, norm=False, wait=True)

move_sig = np.concatenate([convolve_hrtf(sig, fs, ang)
                           for ang in range(-90, 91, 15)], axis=1)
if not building_doc:
    play_sound(move_sig, fs, norm=False, wait=True)

t = np.arange(move_sig.shape[1]) / float(fs)
plt.plot(t, move_sig.T)
plt.xlabel('Time (sec)')
plt.show()
