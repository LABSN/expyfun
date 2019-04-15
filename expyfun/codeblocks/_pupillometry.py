"""Analysis functions (mostly for psychophysics data).
"""

import numpy as np

from ..visual import FixationDot
from ..analyze import sigmoid
from .._utils import logger, verbose_dec
from ..stimuli import window_edges


def _check_pyeparse():
    """Helper to ensure package is available"""
    try:
        import pyeparse  # noqa analysis:ignore
    except ImportError:
        raise ImportError('Cannot run, requires "pyeparse" package')


def _load_raw(el, fname):
    """Helper to load some pupil data"""
    import pyeparse
    fname = el.transfer_remote_file(fname)
    # Load and parse data
    logger.info('Pupillometry: Parsing local file "{0}"'.format(fname))
    raw = pyeparse.RawEDF(fname)
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
    prompt : bool
        If True, a standard prompt message will be displayed.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see expyfun.verbose).

    Returns
    -------
    bgcolor : array
        The background color that maximizes dynamic range.
    fcolor : array
        The corresponding fixation dot color.
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
    if el.recording:
        el.stop()
    el.calibrate()
    if prompt:
        ec.screen_prompt('We will now determine the dynamic '
                         'range of your pupil.\n\n'
                         'Press a button to continue.')
    levels = np.concatenate(([0.], 2 ** np.arange(8) / 255.))
    fixs = levels + 0.2
    n_rep = 2
    # inter-rep interval (allow system to reset)
    iri = 10.0 if not el.dummy_mode else 1.0
    # amount of time between levels
    settle_time = 3.0 if not el.dummy_mode else 0.3
    fix = FixationDot(ec)
    fix.set_colors([fixs[0] * np.ones(3), 'k'])
    ec.set_background_color('k')
    fix.draw()
    ec.flip()
    for ri in range(n_rep):
        ec.wait_secs(iri)
        for ii, (lev, fc) in enumerate(zip(levels, fixs)):
            ec.identify_trial(ec_id='FPDR_%02i' % (ii + 1),
                              el_id=[ii + 1], ttl_id=())
            bgcolor = np.ones(3) * lev
            fcolor = np.ones(3) * fc
            ec.set_background_color(bgcolor)
            fix.set_colors([fcolor, bgcolor])
            fix.draw()
            ec.start_stimulus()
            ec.wait_secs(settle_time)
            ec.check_force_quit()
            ec.stop()
            ec.trial_ok()
        ec.set_background_color('k')
        fix.set_colors([fixs[0] * np.ones(3), 'k'])
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
        assert len(el.file_list) >= 1
        raw, events = _load_raw(el, el.file_list[-1])
        assert len(events) == len(levels) * n_rep
        epochs = pyeparse.Epochs(raw, events, 1, -0.5, settle_time)
        assert len(epochs) == len(levels) * n_rep
        idx = epochs.n_times // 2
        resp = np.median(epochs.get_data('ps')[:, idx:], 1)
    bgcolor = np.mean(resp.reshape((n_rep, len(levels))), 0)
    idx = np.argmin(np.diff(bgcolor)) + 1
    bgcolor = levels[idx] * np.ones(3)
    fcolor = fixs[idx] * np.ones(3)
    logger.info('Pupillometry: optimal background color {0}'.format(bgcolor))
    return bgcolor, fcolor, np.tile(levels, n_rep), resp


def find_pupil_tone_impulse_response(ec, el, bgcolor, fcolor, prompt=True,
                                     verbose=None, targ_is_fm=True):
    """Find pupil impulse response using responses to tones

    Parameters
    ----------
    ec : instance of ExperimentController
        The experiment controller.
    el : instance of EyelinkController
        The Eyelink controller.
    bgcolor : color
        Background color to use.
    fcolor : color
        Fixation dot color to use.
    prompt : bool
        If True, a standard prompt message will be displayed.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see expyfun.verbose).
    targ_is_fm : bool
        If ``True`` then use frequency modulated tones as the target and
        constant frequency tones as the non-target stimuli. Otherwise use
        constant frequency tones are targets and fm tones as non-targets.

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
    if el.recording:
        el.stop()

    #
    # Determine parameters / randomization
    #
    n_stimuli = 300 if not el.dummy_mode else 10
    cal_stim = [0, 75, 150, 225]  # when to offer the subject a break

    delay_range = (3.0, 5.0) if not el.dummy_mode else (0.3, 0.5)
    delay_range = np.array(delay_range)
    targ_prop = 0.25
    stim_dur = 100e-3
    f0 = 1000.  # Hz

    rng = np.random.RandomState(0)
    isis = np.linspace(*delay_range, num=n_stimuli)
    n_targs = int(targ_prop * n_stimuli)
    targs = np.zeros(n_stimuli, bool)
    targs[np.linspace(0, n_stimuli - 1, n_targs + 2)[1:-1].astype(int)] = True
    while(True):  # ensure we randomize but don't start with a target
        idx = rng.permutation(np.arange(n_stimuli))
        isis = isis[idx]
        targs = targs[idx]
        if not targs[0]:
            break

    #
    # Generate stimuli
    #
    fs = ec.stim_fs
    n_samp = int(fs * stim_dur)
    t = np.arange(n_samp).astype(float) / fs
    steady = np.sin(2 * np.pi * f0 * t)
    wobble = np.sin(np.cumsum(f0 + 100 * np.sin(2 * np.pi * (1 / stim_dur) * t)
                              ) / fs * 2 * np.pi)
    std_stim, dev_stim = (steady, wobble) if targ_is_fm else (wobble, steady)
    std_stim = window_edges(std_stim * ec._stim_rms * np.sqrt(2), fs)
    dev_stim = window_edges(dev_stim * ec._stim_rms * np.sqrt(2), fs)

    #
    # Subject "Training"
    #
    ec.stop()
    ec.set_background_color(bgcolor)
    targstr, tonestr = ('wobble', 'beep') if targ_is_fm else ('beep', 'wobble')
    instr = ('Remember to press the button as quickly as possible following '
             'each "{}" sound.\n\nPress the response button to '
             'continue.'.format(targstr))
    if prompt:
        notes = [('We will now determine the response of your pupil to sound '
                  'changes.\n\nYour job is to press the response button '
                  'as quickly as possible when you hear a "{1}" instead '
                  'of a "{0}".\n\nPress a button to hear the "{0}".'
                  ''.format(tonestr, targstr)),
                 ('Now press a button to hear the "{}".'.format(targstr))]
        for text, stim in zip(notes, (std_stim, dev_stim)):
            ec.screen_prompt(text)
            ec.load_buffer(stim)
            ec.wait_secs(0.5)
            ec.play()
            ec.wait_secs(0.5)
            ec.stop()
        ec.screen_prompt(instr)

    fix = FixationDot(ec, colors=[fcolor, bgcolor])
    flip_times = list()
    presses = list()
    assert 0 in cal_stim
    for ii, (isi, targ) in enumerate(zip(isis, targs)):
        if ii in cal_stim:
            if ii != 0:
                el.stop()
                perc = round((100. * ii) / n_stimuli)
                ec.screen_prompt('Great work! You are {0}% done.\n\nFeel '
                                 'free to take a break, then press the '
                                 'button to continue.'.format(perc))
            el.calibrate()
            ec.screen_prompt(instr)
            # let's put the initial color up to allow the system to settle
            fix.draw()
            ec.flip()
            ec.wait_secs(10.0)  # let the pupil settle
        fix.draw()
        ec.load_buffer(dev_stim if targ else std_stim)
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
        raws = list()
        events = list()
        assert len(el.file_list) >= 4
        for fname in el.file_list[-4:]:
            raw, event = _load_raw(el, fname)
            raws.append(raw)
            events.append(event)
        assert sum(len(event) for event in events) == n_stimuli
        epochs = pyeparse.Epochs(raws, events, 1,
                                 tmin=tmin, tmax=delay_range[0])
        response = epochs.pupil_zscores()
        assert response.shape[0] == n_stimuli
        std_err = np.std(response[~targs], axis=0)
        std_err /= np.sqrt(np.sum(~targs))
        response = np.mean(response[~targs], axis=0)
    t = np.arange(len(response)).astype(float) / el.fs + tmin
    return response, t, std_err
