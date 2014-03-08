"""Analysis functions (mostly for psychophysics data).
"""

import numpy as np

from ..visual import FixationDot
from ..analyze import sigmoid, fit_sigmoid
from ..stimuli import repeated_mls, compute_mls_impulse_response
from .._utils import logger, verbose_dec

try:
    import pyeparse
except ImportError:
    pyeparse = None


def _check_pyeparse():
    """Helper to ensure package is available"""
    if pyeparse is None:
        raise ImportError('Cannot run, requires "pyeparse" package')


def _check_fname(el, fname):
    """Helper to deal with Eyelink filename inputs"""
    if fname is None:
        fname = el.start()
    return fname


def _load_raw(el, fname):
    """Helper to load some pupil data"""
    logger.info('Pupillometry: Grabbing remote file "{0}"'.format(fname))
    fname = el.transfer_remote_file(fname)
    # Load and parse data
    logger.info('Pupillometry: Parsing local file "{0}"'.format(fname))
    raw = pyeparse.Raw(fname)
    raw.remove_blink_artifacts()
    events = raw.find_events(raw, 'SYNCTIME')
    return raw, events


@verbose_dec
def find_pupil_dynamic_range(ec, el, settle_time=3.0, fname=None,
                             verbose=None):
    """Find pupil dynamic range

    Parameters
    ----------
    ec : instance of ExperimentController
        The experiment controller.
    el : instance of EyelinkController
        The Eyelink controller.
    isi : float
        Inter-flip interval to use. Should be long enough for the pupil
        to settle (e.g., 3 or 4 seconds).
    fname : str | None
        If str, the filename will be used to process the data from the
        eyelink. If None, a recording will be started using el.start().
    verbose : bool, str, int, or None
        If not None, override default verbose level (see expyfun.verbose).

    Returns
    -------
    lin_reg : array
        The linear range of the pupil response.
    levels : array
        The set of screen levels tested.
    pupil_resp : array
        The pupil response for each level.
    fit_params : tuple
        The four parameters for the sigmoidal fit.

    Notes
    -----
    A four-parameter sigmoid is fit to the data, then the linear portion
    is extracted.
    """
    _check_pyeparse()
    fname = _check_fname(el, fname)
    levels = np.concatenate((np.linspace(0, 1, 10), np.linspace(1, 0, 10)))
    circ = FixationDot(ec, inner_color='k', outer_color='w')
    rect = ec.draw_background_color('k')
    ec.clear_buffer()
    ec.wait_secs(2.0)
    for ii, lev in enumerate(levels):
        ec.identify_trial(ec_id='FPDR_%02i' % (ii + 1),
                          el_id=(ii + 1), ttl_id=())
        rect.set_fill_color(np.ones(3) * lev)
        rect.draw()
        circ.draw()
        ec.flip_and_play()
        ec.wait_secs(settle_time)
        ec.check_force_quit()
    ec.wait_secs(2.0)  # ensure we have enough time
    el.stop()  # stop the recording

    # now we need to parse the data
    if el.dummy_mode:
        pupil_resp = sigmoid(levels, 1, 2, 0.5, 10)
    else:
        # Pull data locally
        raw, events = _load_raw(el, fname)
        assert len(events) == len(levels)
        epochs = pyeparse.Epochs(raw, events, 1, -0.5, settle_time + 1.0)
        assert len(epochs) == len(levels)
        raise NotImplementedError
    fit_params = fit_sigmoid(levels, pupil_resp)
    lower, upper, midpt, slope = fit_params
    logger.info('Pupillometry: Found pupil fit: lower={0}, upper={1}, '
                'midpt={2}, slope={3}'.format(*fit_params))
    lin_reg = np.log([2 - np.sqrt(3), 2 + np.sqrt(3)]) / slope + midpt
    lin_reg = np.clip(lin_reg, 0, 1)
    logger.info('Pupillometry: Linear region: {0}'.format(str(lin_reg)))
    return lin_reg, levels, pupil_resp, fit_params


@verbose_dec
def find_pupil_impulse_response(ec, el, limits=(0.1, 0.9), max_dur=3.0,
                                n_repeats=10, fname=None, verbose=None):
    """Find pupil impulse response

    An MLS sequence will be used, which will be flashy. Be careful!

    Parameters
    ----------
    ec : instance of ExperimentController
        The experiment controller.
    el : instance of EyelinkController
        The Eyelink controller.
    limits : array-like (2 elements)
        Array containing the lower and upper levels (between 0 and 1) to
        use for illumination. Should try to stay within the linear range
        of the pupil response.
    max_dur : float
        Maximum expected duration of the impulse response. If this is too
        short, the tail of the response will wrap to the head.
    fname : str | None
        If str, the filename will be used to process the data from the
        eyelink. If None, a recording will be started using el.start().
    verbose : bool, str, int, or None
        If not None, override default verbose level (see expyfun.verbose).

    Returns
    -------
    prf : array
        The pupil response function.
    screen_fs : float
        The screen refresh rate used to estimate the pupil response.
    """
    _check_pyeparse()
    fname = _check_fname(el, fname)
    limits = np.array(limits).ravel()
    if limits.size != 2:
        raise ValueError('limits must be 2-element array-like')
    if limits.min() < 0 or limits.max() > 1 or limits[0] >= limits[1]:
        raise ValueError('limits must be increasing between 0 and 1')
    logger.info('Pupillometry: Using span {0} to find PRF using MLS'
                ''.format(limits))
    n_repeats = int(n_repeats)
    if n_repeats <= 0:
        raise ValueError('n_repeats must be >= 1, not {0}'.format(n_repeats))
    colors = np.ones((2, 3)) * limits[:, np.newaxis]
    sfs = ec.estimate_screen_fs()

    # let's put the initial color up to allow the system to settle
    ec.clear_buffer()
    rect = ec.draw_background_color(colors[0])
    circ = FixationDot(ec, inner_color='k', outer_color='w')
    circ.draw()
    ec.flip()

    # now let's do some calculations and identify the trial
    ifi = 1. / sfs
    max_samp = int(np.ceil(sfs * max_dur))
    mls, n_resp = repeated_mls(max_samp, n_repeats)  # 0's and 1's
    mls_idx = mls.astype(int)
    n_flip = len(mls)
    ec.identify_trial(ec_id='MLS_{0:0.2f}Hz_{1}samp'.format(sfs, n_flip),
                      el_id=(sfs, n_flip), ttl_id=())
    ec.wait_secs(max_dur * 2)
    flip_times = list()
    for ii, idx in enumerate(mls_idx):
        rect.set_fill_color(colors[idx])
        rect.draw()
        circ.draw()
        if ii == 0:
            flip_times.append(ec.flip_and_play())
        else:
            flip_times.append(ec.flip())
        ec.check_force_quit()

    flip_times = np.array(flip_times)
    if not np.allclose(np.diff(flip_times),
                       ifi * np.ones(len(flip_times) - 1), rtol=0.1):
        raise RuntimeError('Bad flipping')
    el.stop()  # stop the recording

    if el.dummy_mode:
        crf = pyeparse.utils.pupil_kernel(sfs, max_dur)
        response = np.zeros(n_resp)
        response[:len(crf) + len(mls) - 1] = np.convolve(crf, mls)
    else:
        raw, events = _load_raw(el, fname)
        assert len(events) == 1
        dt = np.diff(flip_times[[0, -1]]) / len(flip_times)
        times = np.arange(n_resp) * dt
        response = raw['ps', events[0, 1] + raw.time_as_index(times)]
        assert response.shape == (n_resp,)
    impulse_response = compute_mls_impulse_response(response, mls, n_repeats)
    return impulse_response, sfs
