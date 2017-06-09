# -*- coding: utf-8 -*-
"""File parsing functions
"""

import numpy as np
import csv


def read_tab_raw(fname):
    """Read .tab file from expyfun output without segmenting into trials
    
    Parameters
    ----------
    fname : str
        Input filename.
        
    Returns
    -------
    data : dict
        The data, wach value in the dict being a list of tuples (event, time)
        for each occurrence of that key.
    """
    with open(fname, 'r') as f:
        csvr = csv.reader(f, delimiter='\t')
        lines = [c for c in csvr]

    # first two lines are headers
    assert (len(lines[0]) == 1 and lines[0][0][0] == '#')
    #metadata = ast.literal_eval(lines[0][0][2:])
    assert lines[1] == ['timestamp', 'event', 'value']
    lines = lines[2:]

    header = list(set([l[1] for l in lines]))
    header.sort()

def read_tab(fname, group_start='trial_id', group_end='trial_ok'):
    """Read .tab file from expyfun output

    Parameters
    ----------
    fname : str
        Input filename.
    group_start : str
        Key to use to start a trial/row.
    group_end : str | None
        Key to use to end a trial/row. If None, the next ``group_start``
        will end the current group.

    Returns
    -------
    data : list of dict
        The data, with a dict for each trial. Each value in the dict
        is a list of tuples (event, time) for each occurrence of that
        key.
    """
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
    if group_start not in header:
        raise ValueError('group_start "{0}" not in header: {1}'
                         ''.format(group_start, header))
    if group_end == group_start:
        raise ValueError('group_start cannot equal group_end, use '
                         'group_end=None')
    header = [header.pop(header.index(group_start))] + header
    b1s = np.where([line[1] == group_start for line in lines])[0]
    if group_end is None:
        b2s = np.concatenate((b1s[1:], [len(lines)]))
    else:  # group_end is not None
        if group_end not in header:
            raise ValueError('group_end "{0}" not in header ({1})'
                             ''.format(group_end, header))
        header.append(header.pop(header.index(group_end)))
        b2s = np.where([line[1] == group_end for line in lines])[0]
    if len(b1s) != len(b2s) or not np.all(b1s < b2s):
        raise RuntimeError('bad bounds:\n{0}\n{1}'.format(b1s, b2s))
    data = []
    for b1, b2 in zip(b1s, b2s):
        assert lines[b1][1] == group_start  # prevent stupidity
        if group_end is not None:
            b2 = b2 + 1  # include the end
            assert lines[b2 - 1][1] == group_end
        d = dict()
        these_times = [float(line[0]) for line in lines[b1:b2]]
        these_keys = [line[1] for line in lines[b1:b2]]
        these_vals = [line[2] for line in lines[b1:b2]]
        for ki, key in enumerate(header):
            idx = np.where(key == np.array(these_keys))[0]
            d[key] = [(these_vals[ii], these_times[ii]) for ii in idx]
        data.append(d)
    return data
