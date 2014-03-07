"""Analysis functions (mostly for psychophysics data).
"""

import numpy as np

from ..visual import Circle
from ..analyze import sigmoid, fit_sigmoid
from .._utils import logger, verbose_dec

try:
    import pyeparse
except ImportError:
    pyeparse = None


def _check_pyeparse():
    """Helper to ensure package is available"""
    if pyeparse is None:
        raise ImportError('Cannot run, requires "pyeparse" package')


@verbose_dec
def find_pupil_dynamic_range(ec, el, settle_time=3.0, verbose=None):
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
    verbose : bool, str, int, or None
        If not None, override default verbose level (see expyfun.verbose).

    Returns
    -------
    ranges : array
        The pupil response.

    Notes
    -----
    A four-parameter sigmoid is fit to the data, then the linear portion
    is extracted.
    """
    _check_pyeparse()
    values = np.concatenate((np.linspace(0, 1, 10), np.linspace(1, 0, 10)))
    circ = Circle(ec, 0.1, units='deg', fill_color='k', line_color='w',
                  line_width=1.0)
    rect = ec.draw_background_color('k')
    ec.clear_buffer()
    ec.wait_secs(settle_time)
    for ii, val in enumerate(values):
        ec.identify_trial(ec_id='FPDR_%02i' % (ii + 1),
                          el_id=(ii + 1), ttl_id=())
        rect.set_fill_color(np.ones(3) * val)
        rect.draw()
        circ.draw()
        ec.flip_and_play()
        ec.wait_secs(settle_time)

    # now we need to parse the data
    if el.dummy_mode:
        pupil_levels = sigmoid(values, 1, 2, 0.5, 10)
    else:
        # Load and parse data
        raise NotImplementedError
    lower, upper, midpt, slope = fit_sigmoid(values, pupil_levels)
    logger.info('Pupillometry: Found pupil fit: '
                'lower={0}, upper={1}, midpt={2}, slope={3}'
                ''.format(lower, upper, midpt, slope))
    lin_reg = np.log([2 - np.sqrt(3), 2 + np.sqrt(3)]) / slope + midpt
    lin_reg = np.clip(lin_reg, 0, 1)
    logger.info('Pupillometry: Linear region: {0}'.format(str(lin_reg)))
    return lin_reg


def find_pupil_impulse_response(ec, el, limits=(0.1, 0.9), max_dur=3.0):
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

    Returns
    -------
    prf : array
        The pupil response function.
    """
    limits = np.array(limits).ravel()
    if limits.size != 2:
        raise ValueError('limits must be 2-element array-like')
    if limits.min() < 0 or limits.max() > 1:
        raise ValueError('limits must be between 0 and 1')
    logger.info('Pupillometry: Using span {0} to find PRF using MLS'
                ''.format(limits))
    raise NotImplementedError
