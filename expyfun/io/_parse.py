# -*- coding: utf-8 -*-
"""File parsing functions
"""

import numpy as np
import csv
import ast
import json


def read_tab_raw(fname):
    """Read .tab file from expyfun output without segmenting into trials

    Parameters
    ----------
    fname : str
        Input filename.

    Returns
    -------
    data : dict
        The data with each line from the tab file being a tuple in a list.
    """
    with open(fname, 'r') as f:
        csvr = csv.reader(f, delimiter='\t')
        lines = [c for c in csvr]

    # first two lines are headers
    assert (len(lines[0]) == 1 and lines[0][0][0] == '#')
    #metadata = ast.literal_eval(lines[0][0][2:])
    assert lines[1] == ['timestamp', 'event', 'value']
    lines = lines[2:]

    times = [float(line[0]) for line in lines]
    keys = [line[1] for line in lines]
    vals = [line[2] for line in lines]
    idx = np.arange(len(lines))
    data = [(times[ii], keys[ii], vals[ii]) for ii in idx]
    return data


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
    raw = read_tab_raw(fname)
    lines = [r for r in raw]

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


def reconstruct_tracker(fname):
    """Reconstruct TrackerUD and TrackerBinom objects from .tab files. 

    Parameters
    ----------
    fname : str
        Input filename.

    Returns
    -------
    tr : list of TrackerUD or TrackerBinom
        The tracker objects with all responses such that they are in their
        stopped state (as long as the trackers were allowed to stop during
        the generation of the file.)
    """
    from ..stimuli import TrackerUD, TrackerBinom
    # read in raw data
    raw = read_tab_raw(fname)

    # find tracker_identify and make list of IDs
    tracker_idx = np.where([r[1] == 'tracker_identify' for r in raw])[0]
    if len(tracker_idx) == 0:
        raise ValueError('There are no Trackers in this file.')
    tr = []
    for ii in tracker_idx:
        tracker_id = ast.literal_eval(raw[ii][2])['tracker_id']
        tracker_type = ast.literal_eval(raw[ii][2])['tracker_type']
        # find tracker_ID_init lines and get dict
        init_str = 'tracker_' + str(tracker_id) + '_init'
        tracker_dict_idx = np.where([r[1] == init_str for r in raw])[0][0]
        tracker_dict = json.loads(raw[tracker_dict_idx][2])
        if tracker_type == 'TrackerUD':
            tr.append(TrackerUD(**tracker_dict))
        else:
            tr.append(TrackerBinom(**tracker_dict))
        tr[-1]._tracker_id = tracker_id  # make sure tracker has original ID
        stop_str = 'tracker_' + str(tracker_id) + '_stop'
        tracker_stop_idx = np.where([r[1] == stop_str for r in raw])[0]
        if len(tracker_stop_idx) == 0:
            raise ValueError('Tracker {} has not stopped. All Trackers '
                             'must be stopped.'.format(tracker_id))
        responses = json.loads(raw[tracker_stop_idx[0]][2])['responses']
        # feed in responses from tracker_ID_stop
        [tr[-1].respond(r) for r in responses]
    return tr


def reconstruct_dealer(fname):
    """Reconstruct TrackerDealer object from .tab files. The 
    ``reconstruct_tracker`` function will be called to retrieve the trackers.

    Parameters
    ----------
    fname : str
        Input filename.

    Returns
    -------
    dealer : list of TrackerDealer
        The TrackerDealer objects with all responses such that they are in
        their stopped state.
    """
    from ..stimuli import TrackerDealer
    raw = read_tab_raw(fname)

    # find infor on dealer
    dealer_idx = np.where([r[1] == 'dealer_identify' for r in raw])[0]
    if len(dealer_idx) == 0:
        raise ValueError('There are no TrackerDealers in this file.')
    dealer = []
    for ii in dealer_idx:
        dealer_id = ast.literal_eval(raw[ii][2])['dealer_id']
        dealer_init_str = 'dealer_' + str(dealer_id) + '_init'
        dealer_dict_idx = np.where([r[1] == dealer_init_str 
                                    for r in raw])[0][0]
        dealer_dict = ast.literal_eval(raw[dealer_dict_idx][2])
        dealer_trackers = dealer_dict['trackers']

        # match up tracker objects to id 
        trackers = reconstruct_tracker(fname)
        tr_objects = []
        for t in dealer_trackers:
            idx = np.where([t == t_id._tracker_id for t_id in trackers])[0][0]
            tr_objects.append(trackers[idx])

        # make the dealer object
        max_lag = dealer_dict['max_lag']
        pace_rule = dealer_dict['pace_rule']
        dealer.append(TrackerDealer(None, tr_objects, max_lag, pace_rule))

        # force input responses/log data
        dealer_stop_str = 'dealer_' + str(dealer_id) + '_stop'
        dealer_stop_idx = np.where([r[1] == dealer_stop_str for r in raw])[0]
        if len(dealer_stop_idx) == 0:
            raise ValueError('TrackerDealer {} has not stopped. All dealers '
                             'must be stopped.'.format(tracker_id))
        dealer_stop_log = json.loads(raw[dealer_stop_idx[0]][2])
        log_response_history = dealer_stop_log['response_history']
        log_x_history = dealer_stop_log['x_history']
        log_tracker_history = dealer_stop_log['tracker_history']

        dealer[-1]._response_history = log_response_history
        dealer[-1]._x_history = log_x_history
        dealer[-1]._tracker_history = log_tracker_history
    return dealer
