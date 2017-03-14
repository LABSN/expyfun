# -*- coding: utf-8 -*-
"""
========================
Generate texture stimuli
========================

This shows how to generate texture coherence stimuli.
"""

import numpy as np
import matplotlib.pyplot as plt

from expyfun.stimuli import texture_ERB, play_sound

fs = 24414
n_freqs = 20
n_coh = 18  # very coherent example

# let's make a textured stimilus and play it
sig = texture_ERB(n_freqs, n_coh, fs=fs, seq=('inc', 'nb', 'sam'))
play_sound(sig, fs, norm=True, wait=True)

###############################################################################
# Let's look at the time course
t = np.arange(len(sig)) / float(fs)
fig, ax = plt.subplots(1)
ax.plot(t, sig.T, color='k')
ax.set(xlabel='Time (sec)', ylabel='Amplitude (normalized)', xlim=t[[0, -1]])
fig.tight_layout()

###############################################################################
# And now the spectrogram:
fig, ax = plt.subplots(1, figsize=(8, 2))
img = ax.specgram(sig, NFFT=1024, Fs=fs, noverlap=800)[3]
img.set_clim([img.get_clim()[1] - 50, img.get_clim()[1]])
ax.set(xlim=t[[0, -1]], ylim=[0, 10000], xlabel='Time (sec)',
       ylabel='Freq (Hz)')
fig.tight_layout()

###############################################################################
# And the long-term spectrum:

fig, ax = plt.subplots(1)
ax.psd(sig, NFFT=16384, Fs=fs, color='k')
xticks = [250, 500, 1000, 2000, 4000, 8000]
ax.set(xlabel='Frequency (Hz)', ylabel='Power (dB)', xlim=[100, 10000],
       xscale='log')
ax.set(xticks=xticks)
ax.set(xticklabels=xticks)
fig.tight_layout()
