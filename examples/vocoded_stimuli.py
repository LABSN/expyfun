# -*- coding: utf-8 -*-
"""
========================
Generate vocoded stimuli
========================

This shows how to make simple vocoded stimuli.

@author: larsoner
"""

import numpy as np

from expyfun.stimuli import (get_band_freqs, get_bands, get_env, vocode,
                             play_sound, window_edges, read_wav, rms)
from expyfun import fetch_data_file

data, fs = read_wav(fetch_data_file('audio/dream.wav'))
data = window_edges(data[0], fs)
t = np.arange(data.size) / float(fs)
# noise vocoder
data_noise = vocode(data, fs, mode='noise')
data_noise = data_noise * 0.01 / rms(data_noise)
# sinewave vocoder
data_tone = vocode(data, fs, mode='tone')
data_tone = data_tone * 0.01 / rms(data_tone)
# poisson vocoder
click_rate = 400  # poisson lambda (mean clicks / second)
prob = click_rate / float(fs)
carrier = np.random.choice([0., 1.], data.shape[-1], p=[1 - prob, prob])
edges = get_band_freqs(fs)
bands, filts = get_bands(data, fs, edges, zero_phase=True)
envs, env_filts = zip(*[get_env(x, fs, zero_phase=True) for x in bands])
carrs, carr_filts = get_bands(carrier, fs, edges, zero_phase=True)
data_click = np.zeros_like(data)
for carr, env in zip(carrs, envs):
    data_click += carr * env
data_click = data_click * 0.01 / rms(data_click)

# combine all three
cutoff = data.shape[-1] // 3
data_allthree = data_noise.copy()
data_allthree[cutoff:2 * cutoff] = data_tone[cutoff:2 * cutoff]
data_allthree[2 * cutoff:] = data_click[2 * cutoff:]
snd = play_sound(data_allthree, fs, norm=False, wait=False)

# Uncomment this to play the original, too:
#snd = play_sound(data, fs, norm=False, wait=False)

import matplotlib.pyplot as mpl
mpl.ion()
ax1 = mpl.subplot(2, 1, 1)
ax1.plot(t, data)
ax1.set_title('Original')
ax2 = mpl.subplot(2, 1, 2, sharex=ax1, sharey=ax1)
ax2.plot(t, data_noise)
ax2.set_title('Vocoded')
ax2.set_xlabel('Time (sec)')
