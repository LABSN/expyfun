# -*- coding: utf-8 -*-
"""Functions for using the Coordinate Response Measure (CRM) corpus
"""

import os
from os.path import join
import numpy as np
from ..stimuli import resample
import expyfun.visual as vis
from ._stimuli import window_edges
from ..io import read_wav, write_wav
from fractions import Fraction
from .._utils import fetch_data_file
from zipfile import ZipFile
from joblib import Parallel, delayed, cpu_count
import struct

#from scipy.signal import resample

_fs_binary = 40e3
_sexes = {
    'male': 0,
    'female': 1,
    'm': 0,
    'f': 1,
    0: 0,
    1: 1}
_talker_nums = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    0: 0,
    1: 1,
    2: 2,
    3: 3}
_callsigns = {
    'charlie': 0,
    'ringo': 1,
    'laker': 2,
    'hopper': 3,
    'arrow': 4,
    'tiger': 5,
    'eagle': 6,
    'baron': 7,
    'c': 0,
    'r': 1,
    'l': 2,
    'h': 3,
    'a': 4,
    't': 5,
    'e': 6,
    'b': 7,
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7}
_colors = {
    'blue': 0,
    'red': 1,
    'white': 2,
    'green': 3,
    'b': 0,
    'r': 1,
    'w': 2,
    'g': 3,
    0: 0,
    1: 1,
    2: 2,
    3: 3}
_numbers = {
    'one': 0,
    'two': 1,
    'three': 2,
    'four': 3,
    'five': 4,
    'six': 5,
    'seven': 6,
    'eight': 7,
    '1': 0,
    '2': 1,
    '3': 2,
    '4': 3,
    '5': 4,
    '6': 5,
    '7': 6,
    '8': 7,
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7}

_n_sex = 2
_n_talkers = 4
_n_callsigns = 8
_n_colors = 4
_n_numbers = 8


def _check_parameter_value(param_dict, value, name):
    if isinstance(value, str):
        value = value.lower()
    if value in param_dict.keys():
        return param_dict[value]
    else:
        raise ValueError('{} is not a valid {}. Legal values are: {}'
                         .format(value, name, sorted(param_dict.keys())))


_check_sex = lambda value: _check_parameter_value(_sexes, value, 'sex')
_check_talker_num = lambda value: _check_parameter_value(_talker_nums, value,
                                                         'talker_num')
_check_callsign = lambda value: _check_parameter_value(_callsigns, value,
                                                       'callsign')
_check_number = lambda value: _check_parameter_value(_numbers, value,
                                                     'numbers')
_check_color = lambda value: _check_parameter_value(_colors, value, 'color')


def _read_talker_zip_file(sex, talker_num):
    talker_num_raw = _n_talkers * _sexes[sex] + _talker_nums[talker_num]
    fn = fetch_data_file('crm/Talker %i.zip' % talker_num_raw)
    return ZipFile(fn)


# Read a raw binary CRM file
def _read_binary(zip_file, callsign, color, number,
                 ramp_dur=0.01):
    talk_path = zip_file.filelist[0].orig_filename[:8]
    raw = zip_file.read(join(talk_path, '%02i%02i%02i.BIN' % (
        _callsigns[callsign], _colors[color], _numbers[number])))
    x = np.zeros(len(raw) / 2)
    for bi in np.arange(0, len(raw), 2, dtype=int):
        x[bi / 2] = struct.unpack('<h', raw[bi:bi + 2])[0]
    x /= 16384.
    if ramp_dur:
        return window_edges(x, _fs_binary, dur=ramp_dur)
    else:
        return x


def _pad_zeros(stims, axis=-1, alignment='start', return_array=True):
    """Add zeros to make a list of arrays the same length along a given axis
    """
    if not np.all(np.array([s.ndim for s in stims]) == stims[0].ndim):
        raise(ValueError('All arrays must have the same number of dimensions'))
    lens = np.array([s.shape[axis] for s in stims])
    len_max = lens.max()
    for si, s in enumerate(stims):
        if alignment == 'start':
            n_pre = 0
            n_post = len_max - s.shape[axis]
        elif alignment == 'center':
            n_pre = (len_max - s.shape[axis]) // 2
            n_post = len_max - s.shape[axis] - n_pre
        elif alignment == 'end':
            n_pre = len_max - s.shape[axis]
            n_post = 0
        pre_shape = list(s.shape)
        pre_shape[axis] = n_pre
        post_shape = list(s.shape)
        post_shape[axis] = n_post
        stims[si] = np.concatenate((np.zeros(pre_shape), s,
                                    np.zeros(post_shape)), axis=axis)
    if return_array:
        if not np.all([np.array_equal(s.shape, stims[0].shape)
                       for s in stims]):
            raise(ValueError('Arrays must be the same shape' +
                             'to return an array'))
        return np.array(stims)
    else:
        return stims


def _prepare_stim(zip_file, path_out, sex, tal, cal, col, num, fs_out,
                  fs_binary, rms_binary, ref_rms, n_jobs):
    """Read in a binary CRM file and write out a scaled resampled wav
    """
    x = _read_binary(zip_file, cal, col, num, 0)
    fn = '%i%i%i%i%i.wav' % (sex, tal, cal, col, num)
    #x = resample(x, int((len(x) * fs_out) / fs_binary))  # scipy
    rat = Fraction(np.round(fs_out).astype(int),
                   np.round(fs_binary).astype(int)).limit_denominator(500)
    x = resample(x, rat.numerator, rat.denominator, n_jobs=n_jobs, verbose=0)
    x *= ref_rms / rms_binary
    write_wav(join(path_out, fn), x, fs_out, overwrite=True, verbose=False)


def crm_prepare_corpus(path_out, fs, ref_rms=0.01, talker_list=None,
                       overwrite=False, n_jobs=None, verbose=True):
    """Prepare the CRM corpus for a given sampling rate and convert to wav

    Parameters
    ----------
    path_out : str
        The path to write the prepared CRM corpus.
    fs : int
        The sampling rate of the prepared corpus.
    ref_rms : float
        The baseline RMS value to normalize the stimuli. Default is 0.01.
    talker_list : list of dicts | `None`
        A list of dict objects specifying which talkers to prepare. The
        elements of the dict must be `sex` and `talker_num`, with allowable
        options for sex being `'male'`, `'female'`, `'m'`, `'f'`, or 0 or 1,
        corresponding to male and female, respectively. Valid options for
        `talker_num` are the integers 0 through 3 inclusive. If `None`, all
        talkers of both sexes will be prepared.
    overwrite : bool
        Whether or not to overwrite the files that may already exist in
        `path_out`.
    n_jobs : int | `'cuda'` | `None`
        Number of cores to use. The fastest option, if enabled, is `'cuda'`.
        If `None` it will use all available cores except for one.
    verbose : bool
        Whether or not to ouput status as stimuli are prepared.
    """
    path_out_fs = join(path_out, str(int(fs)))
    rms_binary = 0.099977227591239365  # this doesn't need to be recalculated
    if n_jobs is None:
        n_jobs = cpu_count() - 1
    if n_jobs != 'cuda':
        n_jobs = min([n_jobs, cpu_count()])
    if talker_list is None:
        talker_list = [dict(sex=s, talker_num=t) for s in range(_n_sex) for
                       t in range(_n_talkers)]
    if not os.path.isdir(path_out):
        os.makedirs(path_out)
    if not os.path.isdir(path_out_fs):
        os.makedirs(path_out_fs)
    elif not overwrite:
        raise(RuntimeError('Directory already exists and overwrite=False'))
    cn = [[c, n] for c in range(_n_colors) for n in range(_n_numbers)]

    from time import time
    start_time = time()
    for sex in range(_n_sex):
        if verbose:
            print('Preparing sex %i.' % sex)
        for tal in range(_n_talkers):
            if (dict(sex=_sexes[sex], talker_num=_talker_nums[tal]) in
                    talker_list):
                zf = _read_talker_zip_file(sex, tal)
                if verbose:
                    print('    Preparing talker %i.' % tal),
                if n_jobs != 'cuda':
                    Parallel(n_jobs=n_jobs)(delayed(_prepare_stim)(
                        zf, path_out_fs, sex, tal, cal, col, num, fs,
                        _fs_binary, rms_binary, ref_rms, n_jobs=1) for
                        col, num in cn for cal in range(_n_callsigns))
                else:
                    for cal in range(_n_callsigns):
                        for col, num in cn:
                            _prepare_stim(zf, path_out_fs, sex, tal, cal, col,
                                          num, fs, _fs_binary, rms_binary,
                                          ref_rms, n_jobs='cuda')
                if verbose:
                    print('')
    if verbose:
        print('Finished in %i minutes.' % ((time() - start_time) / 60.))


# Read a CRM wav file that has been prepared for use with expyfun
def crm_sentence(path, fs, sex, talker_num, callsign, color, number,
                 ramp_dur=0.01, stereo=False):
    path = join(path, str(int(fs)))
    if not os.path.isdir(path):
        raise(RuntimeError('prepare_corpus() has not yet been run '
                           'for sampling rate of %i' % fs))
    fn = join(path, '%i%i%i%i%i.wav' % (_sexes[sex], _talker_nums[talker_num],
                                        _callsigns[callsign],
                                        _colors[color], _numbers[number]))
    x = read_wav(fn, verbose=False)[0][0]
    if ramp_dur:
        x = window_edges(x, _fs_binary, dur=ramp_dur)
    if stereo:
        x = np.tile(x[np.newaxis, :], (2, 1))
    return x


def crm_info():
    '''
    Returns lists of options for: sex, talker number, callsign, color, number.
    Example usage: sex, tal, cal, col, num = crm_info()
    '''
    sex = ['male', 'female']
    tal = ['0', '1', '2', '3']
    cal = ['charlie', 'ringo', 'laker', 'hopper',
           'arrow', 'tiger', 'eagle', 'baron']
    col = ['blue', 'red', 'white', 'green']
    num = ['1', '2', '3', '4', '5', '6', '7', '8']
    return sex, tal, cal, col, num


def crm_response_menu(ec, numbers=[1, 2, 3, 4, 5, 6, 7, 8],
                      colors=['blue', 'red', 'white', 'green'],
                      min_wait=0.0, max_wait=np.inf):
    # Set it all up
    mouse_cursor = ec.window._mouse_cursor
    cursor = ec.window.get_system_mouse_cursor(ec.window.CURSOR_HAND)
    ec.window.set_mouse_cursor(cursor)
    colors = [c.lower() for c in colors]
    units = 'norm'
    vert = float(ec.monitor_size_pix[0]) / ec.monitor_size_pix[1]
    h_spacing = 0.1
    v_spacing = h_spacing * vert
    width = h_spacing * 0.8
    height = v_spacing * 0.8
    colors_rgb = {'blue': [0, 0, 1], 'red': [1, 0, 0],
                  'white': [1, 1, 1], 'green': [0, 0.85, 0]}
    n_numbers = len(numbers)
    n_colors = len(colors)
    h_start = -(n_numbers - 1) * h_spacing / 2.
    v_start = (n_colors - 1) * v_spacing / 2.
    font_size = (72 / ec.dpi) * height * 1080 / 2  # same height as box
    h_nudge = h_spacing / 8.
    v_nudge = v_spacing / 12.5

    # Draw the buttons
    rects = []
    for ni, number in enumerate(numbers):
        for ci, color in enumerate(colors):
            pos = [ni * h_spacing + h_start,
                   -ci * v_spacing + v_start,
                   width, height]
            rects += [vis.Rectangle(ec, pos, units=units,
                                    fill_color=colors_rgb[color])]
            rects[-1].draw()
            ec.screen_text(str(number), [pos[0] + h_nudge, pos[1] + v_nudge],
                           color='black',
                           wrap=False, units=units, font_size=font_size)
    ec.flip()
    ec.write_data_line('crm_menu')

    # Get the click
    but = ec.wait_for_click_on(rects, min_wait=min_wait, max_wait=max_wait,
                               live_buttons='left')[1]
    ec.flip()
    ec.window.set_mouse_cursor(mouse_cursor)
    if but is not None:
        sub = np.unravel_index(but, (n_numbers, n_colors))
        resp = [colors[sub[1]], str(numbers[sub[0]])]
        ec.write_data_line('crm_response', resp[0] + ',' + resp[1])
        return resp
    else:
        ec.write_data_line('crm_timeout')
        return (None, None)

#from expyfun import ExperimentController
#with ExperimentController('crm', participant='test', session='test') as ec:
#    for _ in range(3):
#        ec.wait_secs(1)
#        # draw the CRM responder
#        print(crm_response(ec))
