# -*- coding: utf-8 -*-
"""Functions for using the Coordinate Response Measure (CRM) corpus
"""

import os
from os.path import join
import numpy as np
from mne.filter import resample
import expyfun.visual as vis
from ._stimuli import window_edges
from ..io import read_wav, write_wav
from .._utils import fetch_data_file, _get_user_home_path
from zipfile import ZipFile
from joblib import Parallel, delayed, cpu_count
import struct

#from scipy.signal import resample

_fs_binary = 40e3  # the sampling rate of the original corpus binaries
_rms_binary = 0.099977227591239365  # the RMS of the original corpus binaries
_rms_prepped = 0.01  # the RMS for preparation of the whole corpus at an fs

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

_n_sexes = 2
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
    fn = fetch_data_file(join('crm', 'Talker %i.zip' % talker_num_raw))
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


def _prepare_stim(zip_file, path_out, sex, tal, cal, col, num, fs_out, dtype,
                  ref_rms, n_jobs):
    """Read in a binary CRM file and write out a scaled resampled wav
    """
    x = _read_binary(zip_file, cal, col, num, 0)
    fn = '%i%i%i%i%i.wav' % (sex, tal, cal, col, num)
    x = resample(x, fs_out, _fs_binary, n_jobs=n_jobs, verbose=0)
    x *= ref_rms / _rms_binary
    write_wav(join(path_out, fn), x, fs_out, overwrite=True, dtype=dtype,
              verbose=False)


def crm_prepare_corpus(fs, path_out=None, overwrite=False, dtype=np.float64,
                       n_jobs=None, verbose=True):
    """Prepare the CRM corpus for a given sampling rate and convert to wav

    Parameters
    ----------
    fs : int
        The sampling rate of the prepared corpus.
    path_out : str
        The path to write the prepared CRM corpus. In most cases this will be 
        the ``expyfun`` data directory (default), but it allows for other 
        options.
    overwrite : bool
        Whether or not to overwrite the files that may already exist in
        ``path_out``.
    dtype : type
        The data type for saving the data. ``np.float64`` is the default for
        maintaining fidelity. ``np.int16`` is standard for wav files.
    n_jobs : int | ``'cuda'`` | ``None``
        Number of cores to use. The fastest option, if enabled, is ``'cuda'``.
        If ``None`` it will use all available cores except for one.
    verbose : bool
        Whether or not to ouput status as stimuli are prepared.
    """
    if path_out is None:
        path_out = join(_get_user_home_path(), '.expyfun', 'data', 'crm')
    path_out_fs = join(path_out, str(int(fs)))
    if n_jobs is None:
        n_jobs = cpu_count() - 1
    if n_jobs != 'cuda':
        n_jobs = min([n_jobs, cpu_count()])
    if not os.path.isdir(path_out):
        os.makedirs(path_out)
    if not os.path.isdir(path_out_fs):
        os.makedirs(path_out_fs)
    elif not overwrite:
        raise(RuntimeError('Directory already exists and overwrite=False'))
    cn = [[c, n] for c in range(_n_colors) for n in range(_n_numbers)]
    talker_list = [dict(sex=s, talker_num=t) for s in range(_n_sexes) for
                   t in range(_n_talkers)]
    from time import time
    start_time = time()
    for sex in range(_n_sexes):
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
                        zf, path_out_fs, sex, tal, cal, col, num, fs, dtype,
                        _rms_prepped, n_jobs=1) for
                        col, num in cn for cal in range(_n_callsigns))
                else:
                    for cal in range(_n_callsigns):
                        for col, num in cn:
                            _prepare_stim(zf, path_out_fs, sex, tal, cal, col,
                                          num, fs, dtype, _rms_prepped,
                                          n_jobs='cuda')
                if verbose:
                    print('')
    if verbose:
        print('Finished in %i minutes.' % ((time() - start_time) / 60.))


# Read a CRM wav file that has been prepared for use with expyfun
def crm_sentence(fs, sex, talker_num, callsign, color, number, ref_rms=0.01,
                 ramp_dur=0.01, stereo=False, path=None):
    """Get a specific sentence from the hard drive.
        
    Parameters
    ----------
    fs : float
        The sampling rate of the corpus to load. You must have run
        ``prepare_corpus`` for that sampling rate first.
    sex : str | int
        The sex of the talker
    talker_num : str | int
        The zero-indexed talker number. Note that, obviously, male 2 is a
        different talker than female 2.
    callsign : str | int
        The callsign of the sentence.
    color : str | int
        The color of the sentence.
    number : str | int
        The number of the sentence. See Notes below for a cautionary point.
    ref_rms : float
        The baseline RMS value to normalize the stimuli. Default is 0.01.
        Normalization is done at the corpus level, not at the sentence or
        talker.
    ramp_dur : float
        The duration (in seconds) of the onset-offset ramps. Use 0 for no ramp.
    stereo : bool
        Whether to return the data as stereo or not.
    path : str
        The location of the stimulus directory. Defaults to the data directory
        where the raw CRM originals are stored.
    
    Returns
    -------
    sentence : array
        The requested sentence data.
    
    Notes
    -----
    Use ``crm_info`` to see allowable values.
    
    When getting a CRM sentence, you can use the full word (e.g., 'male',
    'green'), the first letter of that word ('m', 'g'), or the index of
    that word in the list of options (0, 3). It should be noted that the
    index of '1' is 0, so care must be taken if using indices for the
    number argument.    
    """
    if path is None:
        path = join(_get_user_home_path(), '.expyfun', 'data', 'crm')
    path = join(path, str(int(fs)))
    if not os.path.isdir(path):
        raise(RuntimeError('prepare_corpus() has not yet been run '
                           'for sampling rate of %i' % fs))
    fn = join(path, '%i%i%i%i%i.wav' % (_sexes[sex], _talker_nums[talker_num],
                                        _callsigns[callsign],
                                        _colors[color], _numbers[number]))
    x = read_wav(fn, verbose=False)[0][0] * ref_rms / _rms_prepped
    if ramp_dur:
        x = window_edges(x, _fs_binary, dur=ramp_dur)
    if stereo:
        x = np.tile(x[np.newaxis, :], (2, 1))
    return x


def crm_info():
    """Get allowable options for CRM stimuli.
    
    Returns
    -------
    options : dict of lists
        Keys are sex, talker number, callsign, color, number.
    """
    sex = ['male', 'female']
    tal = ['0', '1', '2', '3']
    cal = ['charlie', 'ringo', 'laker', 'hopper',
           'arrow', 'tiger', 'eagle', 'baron']
    col = ['blue', 'red', 'white', 'green']
    num = ['1', '2', '3', '4', '5', '6', '7', '8']
    return dict(sex=sex, talker_number=tal, callsign=cal, color=col,
                number=num)


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


class CRMPreload(object):
    """A class that stores the CRM corpus in memory for fast access
    
    Parameters
    ----------
    fs : float
        The sampling rate of the corpus to load. You must have run
        ``prepare_corpus`` for that sampling rate first.
    ref_rms : float
        The baseline RMS value to normalize the stimuli. Default is 0.01.
        Normalization is done at the corpus level, not at the sentence or
        talker.
    ramp_dur : float
        The duration (in seconds) of the onset-offset ramps. Use 0 for no ramp.
    stereo : bool
        Whether to return the data as stereo or not.
    path : str
        The location of the stimulus directory. Defaults to the data directory
        where the raw CRM originals are stored.
    """
    def __init__(self, fs, ref_rms=0.01, ramp_dur=0.01, stereo=False,
                 path=None):
        if path is None:
            path = join(_get_user_home_path(), '.expyfun', 'data', 'crm')
        if not os.path.isdir(path):
            raise(RuntimeError('prepare_corpus() has not yet been run '
                               'for sampling rate of %i' % fs))
        self._all_stim = {sex:{tal:{cal:{col:{num:
                          crm_sentence(fs, sex, tal, cal, col, num, ref_rms,
                                       ramp_dur, stereo, path)
                          for num in range(_n_numbers)}
                          for col in range(_n_colors)}
                          for cal in range(_n_callsigns)}
                          for tal in range(_n_talkers)}
                          for sex in range(_n_sexes)}
    
    def sentence(self, sex, talker_num, callsign, color, number):
        """Get a specific sentence from the pre-loaded data
        
        Parameters
        ----------
        sex : str | int
            The sex of the talker
        talker_num : str | int
            The zero-indexed talker number. Note that, obviously, male 2 is a
            different talker than female 2.
        callsign : str | int
            The callsign of the sentence.
        color : str | int
            The color of the sentence.
        number : str | int
            The number of the sentence. See Notes below for a cautionary point.
            
        Returns
        -------
        sentence : array
            The requested sentence data.
        
        Notes
        -----
        Use ``crm_info`` to see allowable values.
        
        When getting a CRM sentence, you can use the full word (e.g., 'male',
        'green'), the first letter of that word ('m', 'g'), or the index of
        that word in the list of options (0, 3). It should be noted that the
        index of '1' is 0, so care must be taken if using indices for the
        number argument.
        """
        return np.copy(self._all_stim[_sexes[sex]][_talker_nums[talker_num]]
                                     [_callsigns[callsign]][_colors[color]]
                                     [_numbers[number]])
    