# -*- coding: utf-8 -*-
"""
============
Parsing demo
============

This example shows some of the functionality of ``read_tab``.
"""
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

print(__doc__)

from os import path as op
import ast

from expyfun.io import read_tab

fname = op.join(op.dirname(__file__), 'sample.tab')  # from simple_experiment
data = read_tab(fname)
print('Number of trials: %s' % len(data))
keys = list(data[0].keys())
print('Data keys:     %s\n' % keys)
for di, d in enumerate(data):
    if d['trial_id'][0][0] == 'multi-tone':
        print('Trial %s multi-tone' % (di + 1))
        targs = ast.literal_eval(d['multi-tone trial'][0][0])
        presses = [int(k[0]) for k in d['keypress']]
        print('  Targs: %s\n  Press: %s' % (targs, presses))
