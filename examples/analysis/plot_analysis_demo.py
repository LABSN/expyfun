# -*- coding: utf-8 -*-
"""
=============
Analysis demo
=============

This example simulates some 2AFC data and demonstrates the analysis
functions ``dprime_2afc()`` and ``barplot()``.
"""
# Author: Dan McCloy <drmccloy@uw.edu>
#
# License: BSD (3-clause)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import expyfun.analyze as ea

print(__doc__)

# simulate some 2AFC data
trials = 100
c_prob = 0.9
t_prob = 0.6
subjs = ['a', 'b', 'c', 'd', 'e']
ctrl = np.random.binomial(trials, c_prob, len(subjs))
test = np.random.binomial(trials, t_prob, len(subjs))
ctrl_miss = trials - ctrl
test_miss = trials - test
data = pd.DataFrame(dict(ctrl_hit=ctrl, ctrl_miss=ctrl_miss,
                         test_hit=test, test_miss=test_miss), index=subjs)
# calculate dprimes
ctrl_dprime = ea.dprime_2afc(data[['ctrl_hit', 'ctrl_miss']])
test_dprime = ea.dprime_2afc(data[['test_hit', 'test_miss']])
results = pd.DataFrame(dict(ctrl=ctrl_dprime, test=test_dprime))
# plot
subplt, barplt = ea.barplot(results, axis=0, err_bars='sd', lines=True,
                            brackets=[(0, 1)], bracket_text=[r'$p < 10^{-9}$'])
subplt.yaxis.set_label_text('d-prime +/- 1 s.d.')
subplt.set_title('Each line represents a different subject')

# significance brackets example
trials_per_cond = 100
conds = ['ctrl', 'test']
diffs = ['easy', 'hard']
colnames = ['-'.join([x, y]) for x, y in zip(conds * 2,
            np.tile(diffs, (2, 1)).T.ravel().tolist())]
cprob = [0.8, 0.5]
dprob = [0.9, 0.6]
cblock = np.tile(np.atleast_2d(cprob).T, (2, len(subjs))).T
dblock = np.tile(np.atleast_2d(np.repeat(dprob, 2)).T, len(subjs)).T
probs = cblock * dblock
rawscores = np.random.binomial(trials_per_cond, probs, (len(subjs),
                                                        len(conds) *
                                                        len(diffs)))
hitmiss = np.c_[rawscores.ravel(), (trials_per_cond - rawscores).ravel()]
dprimes = ea.dprime_2afc(hitmiss).reshape(rawscores.shape)
results = pd.DataFrame(dprimes, index=subjs, columns=colnames)
subplt, barplt = ea.barplot(results, axis=0, err_bars='sd', lines=True,
                            groups=[(0, 1), (2, 3)], group_names=diffs,
                            bar_names=conds * 2, bracket_group_lines=True,
                            brackets=[(0, 1), (2, 3), (0, 2), (1, 3),
                                      ([0, 1], 3)],  # [2, 3]
                            bracket_text=['foo', 'bar', 'baz', 'snafu',
                                          'foobar'])
subplt.yaxis.set_label_text('d-prime +/- 1 s.d.')
subplt.set_title('Each line represents a different subject')
plt.show()
