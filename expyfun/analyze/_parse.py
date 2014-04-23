# -*- coding: utf-8 -*-
"""File parsing functions
"""

import numpy as np
from os import path as op
import csv


def read_tab(fname, out_fname=None, group_by='trial_id', overwrite=False):
    """Read .tab file from expyfun output

    Parameters
    ----------
    fname : str
        Input filename.
    out_fname : str | None
        Output filename. Can be None if no writing is desired.
    group_by : str
        Tab key to use to group into trials/rows.
    overwrite : bool
        If True, overwrite file (if it exists).

    Returns
    -------
    header : list of str
        The fields in ``data``.
    data : list of lists of lists
        The data, with each row containing a trial, and each column
        containing a type of data.
    """
    if out_fname is not None and op.isfile(out_fname) and not overwrite:
        raise IOError('output filename "{0}" exists, consider using '
                      'overwrite=True'.format(out_fname))
    # load everything into memory for ease of use
    with open(fname, 'r') as f:
        csvr = csv.reader(f, delimiter='\t')
        lines = [c for c in csvr]

    # first two lines are headers
    assert (len(lines[0]) == 1 and lines[0][0][0] == '#')
    #metadata = ast.literal_eval(lines[0][0][2:])
    assert lines[1] == ['timestamp', 'event', 'value']
    lines = lines[2:]

    # determine the event fields
    header = list(set([l[1] for l in lines]))
    header.sort()
    header = [header.pop(header.index(group_by))] + header
    if group_by not in header:
        raise ValueError('group_by "{0}" not in header ({1})'
                         ''.format(group_by, header))
    bounds = [line[1] == group_by for line in lines] + [True]
    bounds = np.where(bounds)[0]
    data = []
    for b1, b2 in zip(bounds[:-1], bounds[1:]):
        assert lines[b1][1] == group_by  # prevent stupidity
        d = [None] * len(header)
        these_times = [line[0] for line in lines[b1:b2]]
        these_keys = [line[1] for line in lines[b1:b2]]
        these_vals = [line[2] for line in lines[b1:b2]]
        for ki, key in enumerate(header):
            idx = np.where(key == np.array(these_keys))[0]
            d[ki] = [(these_vals[ii], these_times[ii]) for ii in idx]
        data.append(d)
    if out_fname is not None:
        with open(out_fname, 'w') as g:
            g.write('\t'.join(header) + '\n')
            for d in data:
                g.write('\t'.join((str(dd) if dd is not None else '')
                                  for dd in data) + '\n')
    return header, data
