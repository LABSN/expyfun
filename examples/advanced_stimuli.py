# -*- coding: utf-8 -*-
"""
=======================================
Generate more advanced auditory stimuli
=======================================

This shows the methods that we provide that facilitate generation
of more advanced stimuli.

@author: larsoner
"""

import numpy as np

from expyfun.stimuli import convolve_hrtf, play_sound, window_edges

fs = 44100
dur = 0.5
freq = 500.
# let's make a square wave
sig = np.sin(freq * 2 * np.pi * np.arange(dur * fs, dtype=float) / fs)
sig = (sig > 0 - 0.5) / 10.  # make it reasonably quiet for play_sound
sig = window_edges(sig, fs)

play_sound(sig, norm=False, wait=True)

move_sig = np.concatenate([convolve_hrtf(sig, fs, ang)
                           for ang in range(-90, 91, 15)], axis=1)
play_sound(move_sig, norm=False, wait=True)
