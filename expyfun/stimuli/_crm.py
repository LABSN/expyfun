"""Functions for using the Coordinate Response Measure (CRM) corpus.
"""

# Author: Ross Maddox <ross.maddox@rochester.edu>
#
# License: BSD (3-clause)

from multiprocessing import cpu_count
import os
from os.path import join
from zipfile import ZipFile

import numpy as np

from ..io import read_wav, write_wav
from .._parallel import parallel_func
from ._stimuli import window_edges
from .. import visual as vis
from .._utils import fetch_data_file, _get_user_home_path

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


def _check(name, value):
    if name.lower() == 'sex':
        param_dict = _sexes
    elif name.lower() == 'talker_num':
        param_dict = _talker_nums
    elif name.lower() == 'callsign':
        param_dict = _callsigns
    elif name.lower() == 'color':
        param_dict = _colors
    elif name.lower() == 'number':
        param_dict = _numbers

    if isinstance(value, str):
        value = value.lower()
    if value in param_dict.keys():
        return param_dict[value]
    else:
        raise ValueError('{} is not a valid {}. Legal values are: {}'
                         .format(value, name,
                                 sorted(k for k in param_dict.keys()
                                        if isinstance(k, int))))


def _get_talker_zip_file(sex, talker_num):
    talker_num_raw = _n_talkers * _sexes[sex] + _talker_nums[talker_num]
    fn = fetch_data_file('crm/Talker%i.zip' % talker_num_raw)
    return fn


# Read a raw binary CRM file
def _read_binary(zip_file, callsign, color, number,
                 ramp_dur=0.01):
    talk_path = zip_file.filelist[0].orig_filename[:8]
    raw = zip_file.read(talk_path + '/%02i%02i%02i.BIN' % (
        _callsigns[callsign], _colors[color], _numbers[number]))
    x = np.fromstring(raw, '<h') / 16384.
    if ramp_dur:
        return window_edges(x, _fs_binary, dur=ramp_dur)
    else:
        return x


def _prepare_stim(zfn, path_out, sex, tal, cal, col, num, fs_out, dtype,
                  ref_rms, n_jobs):
    """Read in a binary CRM file and write out a scaled resampled wav.
    """
    from mne.filter import resample
    zip_file = ZipFile(zfn)
    x = _read_binary(zip_file, cal, col, num, 0)
    fn = '%i%i%i%i%i.wav' % (sex, tal, cal, col, num)
    if int(np.round(fs_out)) != int(np.round(_fs_binary)):
        x = resample(x, fs_out, _fs_binary, n_jobs=n_jobs, verbose=0)
    x *= ref_rms / _rms_binary
    write_wav(join(path_out, fn), x, fs_out, overwrite=True, dtype=dtype,
              verbose=False)


def crm_prepare_corpus(fs, path_out=None, overwrite=False, dtype=np.float64,
                       n_jobs=None, verbose=True, talker_list=None):
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
    talker_list : list of dict
        A list of dicts to define which talkers should be prepared. Each dict
        should have keys ``sex`` and ``talker_num``. Default is to prepare all
        eight talkers (four female, four male), and it is strongly recommended
        to that you do so to avoid headaches. This option is mainly for
        expedient nose tests.
    """
    if path_out is None:
        path_out = join(_get_user_home_path(), '.expyfun', 'data', 'crm')
    if n_jobs is None:
        n_jobs = cpu_count() - 1
    if n_jobs != 'cuda':
        n_jobs = min([n_jobs, cpu_count()])
    if not os.path.isdir(path_out):
        os.makedirs(path_out)

    _crm_prepare_corpus_helper(fs, path_out, overwrite, dtype, n_jobs, verbose,
                               talker_list)


def _crm_prepare_corpus_helper(fs, path_out, overwrite, dtype, n_jobs,
                               verbose, talker_list=None):
    """Helper function that allows prep of one talker for faster testing.
    """
    if talker_list is None:
        talker_list = [dict(sex=s, talker_num=t) for s in range(_n_sexes) for
                       t in range(_n_talkers)]
    else:
        talker_list = [dict(sex=_check('sex', tal['sex']),
                            talker_num=_check('talker_num', tal['talker_num']))
                       for tal in talker_list]
    path_out_fs = join(path_out, str(int(fs)))
    if not os.path.isdir(path_out_fs):
        os.makedirs(path_out_fs)
    elif not overwrite:
        raise RuntimeError('Directory already exists and overwrite=False')

    cn = [[c, n] for c in range(_n_colors) for n in range(_n_numbers)]
    from time import time
    start_time = time()
    for sex in range(_n_sexes):
        if verbose:
            print('Preparing sex %i.' % sex)
        for tal in range(_n_talkers):
            if dict(sex=sex, talker_num=tal) in talker_list:
                zfn = _get_talker_zip_file(sex, tal)
                if verbose:
                    print('    Preparing talker %i.' % tal),
                if n_jobs != 'cuda':
                    parallel, p_fun, _ = parallel_func(_prepare_stim, n_jobs)
                    parallel(p_fun(
                        zfn, path_out_fs, sex, tal, cal, col, num, fs, dtype,
                        _rms_prepped, n_jobs=1) for
                        col, num in cn for cal in range(_n_callsigns))
                else:
                    for cal in range(_n_callsigns):
                        for col, num in cn:
                            _prepare_stim(zfn, path_out_fs, sex, tal, cal, col,
                                          num, fs, dtype, _rms_prepped,
                                          n_jobs='cuda')
                if verbose:
                    print('')
    if verbose:
        print('Finished in %0.1f minutes.' % ((time() - start_time) / 60.))


# Read a CRM wav file that has been prepared for use with expyfun
def crm_sentence(fs, sex, talker_num, callsign, color, number, ref_rms=0.01,
                 ramp_dur=0.01, stereo=False, path=None):
    """Get a specific sentence from the hard drive.

    Parameters
    ----------
    fs : float
        The sampling rate of the corpus to load. You must have run
        :func:`expyfun.stimuli.crm_prepare_corpus` for that sampling
        rate first.
    sex : str | int
        The sex of the talker
    talker_num : str | int
        The zero-indexed talker number. Note that, obviously, male 2 is a
        different talker from female 2.
    callsign : str | int
        The callsign of the sentence.
    color : str | int
        The color of the sentence.
    number : str | int
        The number of the sentence. Note that due to zero-indexing, a value of
        ``'3'`` will cause the talker to say "three", while a value of ``3``
        will cause the talker to say "four" (see Notes below for more
        information).
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

    See Also
    --------
    crm_info
    crm_prepare_corpus

    Notes
    -----
    Use :func:`expyfun.stimuli.crm_info` to see allowable values.

    When getting a CRM sentence, you can use the full word (e.g., 'male',
    'green'), the first letter of that word ('m', 'g'), or the index of
    that word in the list of options (0, 3). It should be noted that the
    index of ``'1'`` is 0, so care must be taken if using indices for the
    number argument.
    """
    if path is None:
        path = join(_get_user_home_path(), '.expyfun', 'data', 'crm')
    path = join(path, str(int(fs)))
    if not os.path.isdir(path):
        raise RuntimeError('prepare_corpus has not yet been run '
                           'for sampling rate of %i' % fs)
    fn = join(path, '%i%i%i%i%i.wav' %
              (_check('sex', sex), _check('talker_num', talker_num),
               _check('callsign', callsign), _check('color', color),
               _check('number', number)))
    if os.path.isfile(fn):
        x = read_wav(fn, verbose=False)[0][0] * ref_rms / _rms_prepped
    else:
        raise RuntimeError('prepare_corpus has not yet been run for the '
                           'requested talker')
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
        Keys are ``['sex', 'talker_number', 'callsign', 'color', 'number']``.
    """
    sex = ['male', 'female']
    tal = ['0', '1', '2', '3']
    cal = ['charlie', 'ringo', 'laker', 'hopper',
           'arrow', 'tiger', 'eagle', 'baron']
    col = ['blue', 'red', 'white', 'green']
    num = ['1', '2', '3', '4', '5', '6', '7', '8']
    return dict(sex=sex, talker_number=tal, callsign=cal, color=col,
                number=num)


def crm_response_menu(ec, colors=['blue', 'red', 'white', 'green'],
                      numbers=['1', '2', '3', '4', '5', '6', '7', '8'],
                      max_wait=np.inf, min_wait=0.0):
    """Create a mouse-driven CRM response menu.

    Parameters
    ----------
    ec : instance of ExperimentController
        Parent EC.
    colors : list
        The colors to include in the menu.
    numbers : list
        The numbers to include in the menu. Note that this follows the same
        conventions as other CRM functions, so that ``'1'`` and ``1`` have
        different behavior.
    max_wait : float
        Duration after which control is returned if no button is clicked.
    min_wait : float
        Duration for which to ignore button clicks.

    Returns
    -------
    Response : tuple
        A tuple containing the color and number selected as ``str``. If the
        menu times out, (None, None) will be returned.
    """
    # Set it all up
    if min_wait > max_wait:
        raise ValueError('min_wait must be <= max_wait')
    start_time = ec.current_time
    mouse_cursor = ec.window._mouse_cursor
    cursor = ec.window.get_system_mouse_cursor(ec.window.CURSOR_HAND)

    colors = [c.lower() for c in colors]
    units = 'norm'
    vert = float(ec.window_size_pix[0]) / ec.window_size_pix[1]
    h_spacing = 0.1
    v_spacing = h_spacing * vert
    width = h_spacing * 0.8
    height = v_spacing * 0.8
    colors_rgb = [[0, 0, 1], [1, 0, 0], [1, 1, 1], [0, 0.85, 0]]
    n_numbers = len(numbers)
    n_colors = len(colors)
    h_start = -(n_numbers - 1) * h_spacing / 2.
    v_start = (n_colors - 1) * v_spacing / 2.
    font_size = (72 / ec.dpi) * height * ec.window_size_pix[1] / 2
    h_nudge = h_spacing / 8.
    v_nudge = v_spacing / 20.

    colors = [_check('color', color) for color in colors]
    numbers = [str(_check('number', number) + 1) for number in numbers]

    if (len(colors) != len(np.unique(colors)) or
            len(numbers) != len(np.unique(numbers))):
        raise ValueError('There can be no repeated colors or numbers in the '
                         'menu.')

    # Draw the buttons
    rects = []
    for ni, number in enumerate(numbers):
        for ci, color in enumerate(colors):
            pos = [ni * h_spacing + h_start,
                   -ci * v_spacing + v_start,
                   width, height]
            rects += [vis.Rectangle(
                ec, pos, units=units,
                fill_color=colors_rgb[color])]
            rects[-1].draw()
            ec.screen_text(number, [pos[0] + h_nudge, pos[1] + v_nudge],
                           color='black',
                           wrap=False, units=units, font_size=font_size)
    ec.flip()
    ec.write_data_line('crm_menu')

    # Wait for min_wait and get the click
    while ec.current_time - start_time < min_wait:
        ec.check_force_quit()
    ec.window.set_mouse_cursor(cursor)
    max_wait = np.maximum(0, max_wait - (ec.current_time - start_time))
    but = ec.wait_for_click_on(rects, max_wait=max_wait,
                               live_buttons='left')[1]
    ec.flip()
    ec.window.set_mouse_cursor(mouse_cursor)
    if but is not None:
        sub = np.unravel_index(but, (n_numbers, n_colors))
        resp = ('brwg'[colors[sub[1]]], numbers[sub[0]])
        ec.write_data_line('crm_response', resp[0] + ',' + resp[1])
        return resp
    else:
        ec.write_data_line('crm_timeout')
        return (None, None)


class CRMPreload(object):
    """Store the CRM corpus in memory for fast access.

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
        if not os.path.isdir(join(path, str(fs))):
            raise RuntimeError('prepare_corpus has not yet been run '
                               'for sampling rate of %i' % fs)
        self._excluded = []
        self._all_stim = {}
        for sex in range(_n_sexes):
            for tal in range(_n_talkers):
                for cal in range(_n_callsigns):
                    for col in range(_n_colors):
                        for num in range(_n_numbers):
                            stim_id = '%i%i%i%i%i' % (sex, tal, cal, col, num)
                            try:
                                self._all_stim[stim_id] = \
                                    crm_sentence(fs, sex, tal, cal, col, num,
                                                 ref_rms, ramp_dur, stereo,
                                                 path)
                            except Exception:
                                self._excluded += [stim_id]

    def sentence(self, sex, talker_num, callsign, color, number):
        """Get a specific sentence from the pre-loaded data.

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
            The number of the sentence. Note that due to zero-indexing, a value
            of ``'3'`` will cause the talker to say "three", while a value of
            ``3`` will cause the talker to say "four" (see Notes below for more
            information).

        Returns
        -------
        sentence : array
            The requested sentence data.

        See Also
        --------
        expyfun.stimuli.crm_info

        Notes
        -----
        Use :func:`expyfun.stimuli.crm_info` to see allowable values.

        When getting a CRM sentence, you can use the full word (e.g., 'male',
        'green'), the first letter of that word ('m', 'g'), or the index of
        that word in the list of options (0, 3). It should be noted that the
        index of ``'1'`` is 0, so care must be taken if using indices for the
        number argument.
        """
        stim_id = '%i%i%i%i%i' % (
            _check('sex', sex), _check('talker_num', talker_num),
            _check('callsign', callsign), _check('color', color),
            _check('number', number))
        if stim_id in self._excluded:
            raise RuntimeError('prepare_corpus has not yet been run for the '
                               'requested talker')
        return self._all_stim[stim_id].copy()
