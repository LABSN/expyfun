"""Analysis functions (mostly for psychophysics data).
"""

import numpy as np
from scipy import signal

from ..visual import FixationDot
from ..analyze import sigmoid
from .._utils import logger, verbose_dec


def _check_pyeparse():
    """Helper to ensure package is available"""
    try:
        import pyeparse  # noqa
    except ImportError:
        raise ImportError('Cannot run, requires "pyeparse" package')


def _check_fname(el):
    """Helper to deal with Eyelink filename inputs"""
    if not el._is_file_open:
        fname = el._open_file()
    else:
        fname = el._current_open_file
    if not el.recording:
        el._start_recording()
    return fname


def _load_raw(el, fname):
    """Helper to load some pupil data"""
    import pyeparse
    fname = el.transfer_remote_file(fname)
    # Load and parse data
    logger.info('Pupillometry: Parsing local file "{0}"'.format(fname))
    raw = pyeparse.Raw(fname)
    raw.remove_blink_artifacts()
    events = raw.find_events('SYNCTIME', 1)
    return raw, events


@verbose_dec
def find_pupil_dynamic_range(ec, el, prompt=True, verbose=None):
    """Find pupil dynamic range

    Parameters
    ----------
    ec : instance of ExperimentController
        The experiment controller.
    el : instance of EyelinkController
        The Eyelink controller.
    fname : str | None
        If str, the filename will be used to process the data from the
        eyelink. If None, a recording will be started.
    prompt : bool
        If True, a standard prompt message will be displayed.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see expyfun.verbose).

    Returns
    -------
    bgcolor : array
        The background color that maximizes dynamic range.
    levels : array
        The levels shown.
    responses : array
        The average responses to each level.

    Notes
    -----
    If ``el.dummy_mode`` is on, the test will run at around 10x the speed.
    """
    _check_pyeparse()
    import pyeparse
    if prompt:
        ec.screen_prompt('We will now determine the dynamic '
                         'range of your pupil.<br><br>'
                         'Press a button to continue.')
    fname = _check_fname(el)
    levels = np.concatenate(([0.], 2 ** np.arange(8) / 255.))
    n_rep = 2
    # inter-rep interval (allow system to reset)
    iri = 10.0 if not el.dummy_mode else 1.0
    # amount of time between levels
    settle_time = 3.0 if not el.dummy_mode else 0.3
    fix = FixationDot(ec)
    bgrect = ec.draw_background_color('k')
    fix.draw()
    ec.flip()
    ec.clear_buffer()
    for ri in range(n_rep):
        ec.wait_secs(iri)
        for ii, lev in enumerate(levels):
            ec.identify_trial(ec_id='FPDR_%02i' % (ii + 1),
                              el_id=[ii + 1], ttl_id=())
            bgrect.set_fill_color(np.ones(3) * lev)
            bgrect.draw()
            fix.draw()
            ec.start_stimulus()
            ec.wait_secs(settle_time)
            ec.check_force_quit()
            ec.trial_ok()
        bgrect.set_fill_color('k')
        bgrect.draw()
        fix.draw()
        ec.flip()
    el.stop()  # stop the recording
    ec.screen_prompt('Processing data, please wait...', max_wait=0,
                     clear_after=False)

    # now we need to parse the data
    if el.dummy_mode:
        resp = sigmoid(np.tile(levels, n_rep), 1000, 3000, 0.01, -100)
        resp += np.random.rand(*resp.shape) * 500 - 250
    else:
        # Pull data locally
        raw, events = _load_raw(el, fname)
        assert len(events) == len(levels) * n_rep
        epochs = pyeparse.Epochs(raw, events, 1, -0.5, settle_time)
        assert len(epochs) == len(levels) * n_rep
        idx = epochs.n_times // 2
        resp = np.median(epochs.get_data('ps')[:, idx:], 1)
    bgcolor = np.mean(resp.reshape((n_rep, len(levels))), 0)
    bgcolor = levels[np.argmin(np.diff(bgcolor)) + 1] * np.ones(3)
    logger.info('Pupillometry: optimal background color {0}'.format(bgcolor))
    return bgcolor, np.tile(levels, n_rep), resp


def find_pupil_tone_impulse_response(ec, el, bgcolor, prompt=True,
                                     verbose=None):
    """Find pupil impulse response using responses to tones

    Parameters
    ----------
    ec : instance of ExperimentController
        The experiment controller.
    el : instance of EyelinkController
        The Eyelink controller.
    bgcolor : color
        Background color to use.
    prompt : bool
        If True, a standard prompt message will be displayed.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see expyfun.verbose).

    Returns
    -------
    srf : array
        The pupil response function to sound.
    t : array
        The time points for the response function.
    std_err : array
        The standard error as a function of time.

    Notes
    -----
    If ``el.dummy_mode`` is on, the test will run at around 10x the speed.
    """
    _check_pyeparse()
    import pyeparse

    # let's do some calculations
    n_stimuli = 125 if not el.dummy_mode else 10
    delay_range = (3.0, 4.0) if not el.dummy_mode else (0.3, 0.4)
    delay_range = np.array(delay_range)
    targ_prop = 0.2
    stim_dur = 50e-3
    f0 = 500.  # Hz

    rng = np.random.RandomState(1)
    isis = (rng.rand(n_stimuli) * np.diff(delay_range)
            + delay_range[0])
    targs = np.zeros(n_stimuli, bool)
    n_targs = int(targ_prop * n_stimuli)
    while(True):  # ensure no two targets in a row
        idx = np.sort(rng.permutation(np.arange(n_stimuli))[:n_targs])
        if (not np.any(np.diff(idx) == 1)) and idx[0] != 0:
            targs[idx] = True
            break

    # generate stimuli
    fs = ec.stim_fs
    n_samp = int(fs * stim_dur)
    window = signal.windows.hanning(int(0.01 * fs))
    idx = len(window) // 2
    window = np.concatenate((window[:idx + 1], np.ones(n_samp - 2 * idx - 2),
                             window[idx:]))
    freqs = np.ones(n_samp) * f0
    t = np.arange(n_samp).astype(float) / fs
    tone_stim = np.sin(2 * np.pi * freqs * t)
    freqs = 100 * np.sin(2 * np.pi * (1 / stim_dur) * t) + f0
    sweep_stim = np.sin(2 * np.pi * np.cumsum(freqs) / fs)
    tone_stim *= (ec._stim_rms * np.sqrt(2)) * window
    sweep_stim *= (ec._stim_rms * np.sqrt(2)) * window

    ec.stop()
    ec.clear_buffer()
    if prompt:
        ec.screen_prompt('We will now determine the response of your pupil '
                         'to sound changes.<br><br>Your job is to press the'
                         'repsonse button as quickly as possible when you '
                         'hear a "wobble" instead of a "beep".<br><br>'
                         'Press a button to hear the "beep".')
        ec.load_buffer(tone_stim)
        ec.wait_secs(0.5)
        ec.play()
        ec.wait_secs(0.5)
        ec.stop()
        ec.screen_prompt('Now press a button to hear the "wobble".')
        ec.load_buffer(sweep_stim)
        ec.wait_secs(0.5)
        ec.play()
        ec.wait_secs(0.5)
        ec.stop()
        ec.screen_prompt('Remember to press the button as quickly as '
                         'possible following each "wobble" sound.<br><br>'
                         'Press the response button to continue.')
    fname = _check_fname(el)

    # let's put the initial color up to allow the system to settle
    bgrect = ec.draw_background_color(bgcolor)
    fix = FixationDot(ec)
    fix.draw()
    ec.flip()

    ec.wait_secs(3.0)
    flip_times = list()
    presses = list()
    for ii, (isi, targ) in enumerate(zip(isis, targs)):
        bgrect.draw()
        fix.draw()
        ec.load_buffer(sweep_stim if targ else tone_stim)
        ec.identify_trial(ec_id='TONE_{0}'.format(int(targ)),
                          el_id=[int(targ)], ttl_id=[int(targ)])
        flip_times.append(ec.start_stimulus())
        presses.append(ec.wait_for_presses(isi))
        ec.stop()
        ec.trial_ok()
    el.stop()  # stop the recording
    ec.screen_prompt('Processing data, please wait...', max_wait=0,
                     clear_after=False)

    flip_times = np.array(flip_times)
    tmin = -0.5
    if el.dummy_mode:
        pk = pyeparse.utils.pupil_kernel(el.fs, 3.0 - tmin)
        response = np.zeros(len(pk))
        offset = int(el.fs * 0.5)
        response[offset:] = pk[:-offset]
        std_err = np.ones_like(response) * 0.1 * response.max()
        std_err += np.random.rand(std_err.size) * 0.1 * response.max()
    else:
        raw, events = _load_raw(el, fname)
        assert len(events) == n_stimuli
        epochs = pyeparse.Epochs(raw, events, 1,
                                 tmin=tmin, tmax=delay_range[0])
        response = epochs.pupil_zscores()
        assert response.shape[0] == n_stimuli
        std_err = np.std(response[~targs], axis=0)
        std_err /= np.sqrt(np.sum(~targs))
        response = np.mean(response[~targs], axis=0)
    t = np.arange(len(response)).astype(float) / el.fs + tmin
    return response, t, std_err
