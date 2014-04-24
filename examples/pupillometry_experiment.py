"""
==========================================
Experiment using eye-tracking pupillometry
==========================================

Integration with Eyelink functionality makes programming experiments
using eye-tracking simpler.
"""
# Author: Eric Larson <larsoner@uw.edu>
#
# License: BSD (3-clause)

print(__doc__)

import numpy as np
from expyfun import ExperimentController, EyelinkController
from expyfun.codeblocks import (find_pupil_dynamic_range,
                                find_pupil_tone_impulse_response)


with ExperimentController('pupilExp', full_screen=True, participant='foo',
                          session='001', output_dir=None) as ec:
    el = EyelinkController(ec)
    bgcolor, fcolor, lev, resp = find_pupil_dynamic_range(ec, el)
    prf, t_srf, e_prf = find_pupil_tone_impulse_response(ec, el, bgcolor,
                                                         fcolor)

import matplotlib.pyplot as plt
plt.ion()

uni_lev = np.unique(lev)
uni_lev_label = (255 * uni_lev).astype(int)
uni_lev[uni_lev == 0] = np.sort(uni_lev)[1] / 2.
r = resp.reshape((len(lev) // len(uni_lev), len(uni_lev)))
r_span = [r.min(), r.max()]
# Grayscale responses
ax = plt.subplot(2, 1, 1, xlabel='Screen level', ylabel='Pupil dilation (AU)')
ax.plot([bgcolor, bgcolor], r_span, linestyle='--', color='r')
ax.fill_between(uni_lev, np.min(r, 0), np.max(r, 0), facecolor=(1, 1, 0),
                edgecolor='none')
ax.semilogx(uni_lev, np.mean(r, 0), color='k')
ax.set_xlim(uni_lev[[0, -1]])
ax.set_ylim(r_span)
plt.xticks(uni_lev, uni_lev_label)

# PRF
ax = plt.subplot(2, 1, 2, xlabel='Time (s)', ylabel='Pupil response (AU)')
ax.fill_between(t_srf, prf - e_prf, prf + e_prf, facecolor=(1, 1, 0),
                edgecolor='none')
ax.plot(t_srf, prf, color='k')
ax.set_xlim(t_srf[[0, -1]])
plt.tight_layout()
