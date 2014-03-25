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
    iri = 10.0  # inter-rep interval (allow system to reset)
    settle_time = 3.0  # amount of time between levels
    fix = FixationDot(ec)
    bgrect = ec.draw_background_color('k')
    fix.draw()
    ec.flip()
    ec.clear_buffer()
    for ri in range(n_rep):
        ec.wait_secs(iri)
        for ii, lev in enumerate(levels):
            ec.identify_trial(ec_id='FPDR_%02i' % (ii + 1),
                              el_id=(ii + 1), ttl_id=())
            bgrect.set_fill_color(np.ones(3) * lev)
            bgrect.draw()
            fix.draw()
            ec.flip_and_play()
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
        resp = sigmoid(levels, 1, 2, 0.5, 10)
    else:
        # Pull data locally
        raw, events = _load_raw(el, fname)
        assert len(events) == len(levels) * n_rep
        epochs = pyeparse.Epochs(raw, events, 1, -0.5, settle_time)
        assert len(epochs) == len(levels) * n_rep
        idx = epochs.n_times // 2
        resp = np.median(epochs.get_data('ps')[:, idx:], 1)
        resp = np.mean(resp.reshape((n_rep, len(levels))), 0)
    bgcolor = levels[np.argmin(np.diff(resp))] * np.ones(3)
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
    """
    _check_pyeparse()
    import pyeparse
    if prompt:
        ec.screen_prompt('We will now determine the response of your pupil '
                         'to sound changes.<br><br>Count the number of times '
                         'you hear the tone "wobble" instead of staying '
                         'constant.<br><br>Press a button to continue.')
    fname = _check_fname(el)

    # let's put the initial color up to allow the system to settle
    ec.clear_buffer()
    bgrect = ec.draw_background_color(bgcolor)
    fix = FixationDot(ec)
    fix.draw()
    ec.flip()

    # now let's do some calculations
    n_stimuli = 125
    delay_range = np.array((3.0, 4.0))
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

    ec.wait_secs(3.0)
    flip_times = list()
    presses = list()
    for ii, (isi, targ) in enumerate(zip(isis, targs)):
        bgrect.draw()
        fix.draw()
        ec.load_buffer(sweep_stim if targ else tone_stim)
        ec.identify_trial(ec_id='TONE_{0}'.format(int(targ)),
                          el_id=[int(targ)], ttl_id=[int(targ)])
        flip_times.append(ec.flip_and_play())
        presses.append(ec.wait_for_presses(isi))
        ec.stop()
        ec.trial_ok()
    el.stop()  # stop the recording
    ec.screen_prompt('Processing data, please wait...', max_wait=0,
                     clear_after=False)

    flip_times = np.array(flip_times)
    tmin = -0.5
    if el.dummy_mode:
        pk = pyeparse.utils.pupil_kernel(el.fs, delay_range[0] - tmin)
        response = np.zeros(len(pk))
        offset = int(el.fs * 0.5)
        response[offset:] = pk[:-offset]
    else:
        raw, events = _load_raw(el, fname)
        assert len(events) == n_stimuli
        epochs = pyeparse.Epochs(raw, events, 1,
                                 tmin=tmin, tmax=delay_range[0])
        response = np.reshape(epochs.pupil_zscores())
        assert response.shape[0] == n_stimuli
        response = np.mean(response[~targs], axis=0)
    t = np.arange(len(response)).astype(float) / el.fs + tmin
    return response, t
