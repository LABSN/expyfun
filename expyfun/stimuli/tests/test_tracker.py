import numpy as np

from expyfun.stimuli import TrackerUD, TrackerBinom, TrackerDealer, TrackerMHW
from expyfun import ExperimentController
import pytest
from numpy.testing import assert_equal
from expyfun._utils import requires_opengl21


def callback(event_type, value=None, timestamp=None):
    """Callback."""
    print(event_type, value, timestamp)


std_kwargs = dict(output_dir=None, full_screen=False, window_size=(1, 1),
                  participant='foo', session='01', stim_db=0.0, noise_db=0.0,
                  trigger_controller='dummy', response_device='keyboard',
                  audio_controller='sound_card',
                  verbose=True, version='dev')


@pytest.mark.timeout(15)
@requires_opengl21
def test_tracker_ud(hide_window):
    """Test TrackerUD"""
    import matplotlib.pyplot as plt
    tr = TrackerUD(callback, 3, 1, 1, 1, np.inf, 10, 1)
    with ExperimentController('test', **std_kwargs) as ec:
        tr = TrackerUD(ec, 3, 1, 1, 1, np.inf, 10, 1)
    tr = TrackerUD(None, 3, 1, 1, 1, 10, np.inf, 1)
    rand = np.random.RandomState(0)
    while not tr.stopped:
        tr.respond(rand.rand() < tr.x_current)
    assert_equal(tr.n_reversals, tr.stop_reversals)

    tr = TrackerUD(None, 3, 1, 1, 1, np.inf, 10, 1)
    tr.threshold()
    rand = np.random.RandomState(0)
    while not tr.stopped:
        tr.respond(rand.rand() < tr.x_current)
    # test responding after stopped
    pytest.raises(RuntimeError, tr.respond, 0)

    # all the properties better work
    tr.up
    tr.down
    tr.step_size_up
    tr.step_size_down
    tr.stop_reversals
    tr.stop_trials
    tr.start_value
    tr.x_min
    tr.x_max
    tr.stopped
    tr.x
    tr.responses
    tr.n_trials
    tr.n_reversals
    tr.reversals
    tr.reversal_inds
    fig, ax, lines = tr.plot()
    tr.plot_thresh(ax=ax)
    tr.plot_thresh()
    plt.close(fig)
    ax = plt.axes()
    fig, ax, lines = tr.plot(ax)
    plt.close(fig)
    tr.threshold()
    tr.check_valid(2)

    # bad callback type
    pytest.raises(TypeError, TrackerUD, 'foo', 3, 1, 1, 1, 10, np.inf, 1)

    # test dynamic step size and error conditions
    tr = TrackerUD(None, 3, 1, [1, 0.5], [1, 0.5], 10, np.inf, 1,
                   change_indices=[2])
    tr.respond(True)

    tr = TrackerUD(None, 1, 1, 0.75, 0.75, np.inf, 9, 1,
                   x_min=0, x_max=2)
    responses = [True, True, True, False, False, False, False, True, False]
    with pytest.warns(UserWarning, match='exceeded x_min'):
        for r in responses:  # run long enough to encounter change_indices
            tr.respond(r)
    assert(tr.check_valid(1))  # make sure checking validity is good
    assert(not tr.check_valid(3))
    pytest.raises(ValueError, tr.threshold, 1)
    tr.threshold(3)
    assert_equal(tr.n_trials, tr.stop_trials)

    # run tests with ignore too--should generate warnings, but no error
    tr = TrackerUD(None, 1, 1, 0.75, 0.25, np.inf, 8, 1,
                   x_min=0, x_max=2, repeat_limit='ignore')
    responses = [False, True, False, False, True, True, False, True]
    with pytest.warns(UserWarning, match='exceeded x_min'):
        for r in responses:  # run long enough to encounter change_indices
            tr.respond(r)
    tr.threshold(0)

    # bad stop_trials
    pytest.raises(ValueError, TrackerUD, None, 3, 1, 1, 1, 10, 'foo', 1)

    # bad stop_reversals
    pytest.raises(ValueError, TrackerUD, None, 3, 1, 1, 1, 'foo', 10, 1)

    # change_indices too long
    pytest.raises(ValueError, TrackerUD, None, 3, 1, [1, 0.5], [1, 0.5], 10,
                  np.inf, 1, change_indices=[1, 2])
    # step_size_up length mismatch
    pytest.raises(ValueError, TrackerUD, None, 3, 1, [1], [1, 0.5], 10,
                  np.inf, 1, change_indices=[2])
    # step_size_down length mismatch
    pytest.raises(ValueError, TrackerUD, None, 3, 1, [1, 0.5], [1], 10,
                  np.inf, 1, change_indices=[2])
    # bad change_rule
    pytest.raises(ValueError, TrackerUD, None, 3, 1, [1, 0.5], [1, 0.5], 10,
                  np.inf, 1, change_indices=[2], change_rule='foo')
    # no change_indices (i.e. change_indices=None)
    pytest.raises(ValueError, TrackerUD, None, 3, 1, [1, 0.5], [1, 0.5], 10,
                  np.inf, 1)

    # start_value scalar type checking
    pytest.raises(TypeError, TrackerUD, None, 3, 1, [1, 0.5], [1, 0.5], 10,
                  np.inf, [9, 5], change_indices=[2])
    pytest.raises(TypeError, TrackerUD, None, 3, 1, [1, 0.5], [1, 0.5], 10,
                  np.inf, None, change_indices=[2])

    # test with multiple change_indices
    tr = TrackerUD(None, 3, 1, [3, 2, 1], [3, 2, 1], 10, np.inf, 1,
                   change_indices=[2, 4], change_rule='reversals')


@requires_opengl21
def test_tracker_binom(hide_window):
    """Test TrackerBinom"""
    tr = TrackerBinom(callback, 0.05, 0.1, 5)
    with ExperimentController('test', **std_kwargs) as ec:
        tr = TrackerBinom(ec, 0.05, 0.1, 5)
    tr = TrackerBinom(None, 0.05, 0.5, 2, stop_early=False)
    while not tr.stopped:
        tr.respond(False)
    assert(tr.n_trials == 2)
    assert(not tr.success)

    tr = TrackerBinom(None, 0.05, 0.5, 1000)
    while not tr.stopped:
        tr.respond(True)

    tr = TrackerBinom(None, 0.05, 0.5, 1000, 100)
    while not tr.stopped:
        tr.respond(True)
    assert(tr.n_trials == 100)

    tr.alpha
    tr.chance
    tr.max_trials
    tr.stop_early
    tr.p_val
    tr.min_p_val
    tr.max_p_val
    tr.n_trials
    tr.n_wrong
    tr.n_correct
    tr.pc
    tr.responses
    tr.stopped
    tr.success
    tr.x_current
    tr.x
    tr.stop_rule


def test_tracker_dealer():
    """Test TrackerDealer."""
    # test TrackerDealer with TrackerUD
    trackers = [[TrackerUD(None, 1, 1, 0.06, 0.02, 20, np.inf,
                           1) for _ in range(2)] for _ in range(3)]
    dealer_ud = TrackerDealer(callback, trackers)

    # can't respond to a trial twice
    dealer_ud.next()
    dealer_ud.respond(True)
    pytest.raises(RuntimeError, dealer_ud.respond, True)

    dealer_ud = TrackerDealer(callback, np.array(trackers))

    # can't respond before you pick a tracker and get a trial
    pytest.raises(RuntimeError, dealer_ud.respond, True)

    rand = np.random.RandomState(0)
    for sub, x_current in dealer_ud:
        dealer_ud.respond(rand.rand() < x_current)
        assert(np.abs(dealer_ud.trackers[0, 0].n_reversals -
                      dealer_ud.trackers[1, 0].n_reversals) <= 1)

    # test array-like indexing
    dealer_ud.trackers[0]
    dealer_ud.trackers[:]
    dealer_ud.trackers[[1, 0, 1]]
    [d for d in dealer_ud.trackers]
    dealer_ud.shape
    dealer_ud.history()
    dealer_ud.history(True)

    # bad rand type
    trackers = [TrackerUD(None, 1, 1, 0.06, 0.02, 20, 50, 1)
                for _ in range(2)]
    pytest.raises(TypeError, TrackerDealer, trackers, rand=1)

    # test TrackerDealer with TrackerBinom
    trackers = [TrackerBinom(None, 0.05, 0.5, 50, stop_early=False)
                for _ in range(2)]    # start_value scalar type checking
    pytest.raises(TypeError, TrackerUD, None, 3, 1, [1, 0.5], [1, 0.5], 10,
                  np.inf, [9, 5], change_indices=[2])
    dealer_binom = TrackerDealer(callback, trackers, pace_rule='trials')
    for sub, x_current in dealer_binom:
        dealer_binom.respond(True)

    # if you're dealing from TrackerBinom, you can't use stop_early feature
    trackers = [TrackerBinom(None, 0.05, 0.5, 50, stop_early=True, x_current=3)
                for _ in range(2)]
    pytest.raises(ValueError, TrackerDealer, callback, trackers, 1, 'trials')

    # if you're dealing from TrackerBinom, you can't use reversals to pace
    pytest.raises(ValueError, TrackerDealer, callback, trackers, 1)


def test_tracker_mhw(hide_window):
    """Test TrackerMHW"""
    import matplotlib.pyplot as plt
    tr = TrackerMHW(callback, 0, 120)
    with ExperimentController('test', **std_kwargs) as ec:
        tr = TrackerMHW(ec, 0, 120)
    tr = TrackerMHW(None, 0, 120)
    rand = np.random.RandomState(0)
    while not tr.stopped:
        tr.respond(int(rand.rand() * 100) < tr.x_current)

    tr = TrackerMHW(None, 0, 120)
    rand = np.random.RandomState(0)
    while not tr.stopped:
        tr.respond(int(rand.rand() * 100) < tr.x_current)
        assert(tr.check_valid(1))  # make sure checking validity is good
    # test responding after stopped
    pytest.raises(RuntimeError, tr.respond, 0)

    for key in ('base_step', 'factor_down', 'factor_up_nr', 'start_value',
                'x_min', 'x_max', 'n_up_stop', 'repeat_limit',
                'n_correct_levels', 'threshold', 'stopped', 'x', 'x_current',
                'responses', 'n_trials', 'n_reversals', 'reversals',
                'reversal_inds', 'threshold_reached'):
        assert hasattr(tr, key)

    fig, ax, lines = tr.plot()
    tr.plot_thresh(ax=ax)
    tr.plot_thresh()
    plt.close(fig)
    ax = plt.axes()
    fig, ax, lines = tr.plot(ax)
    plt.close(fig)

    # start_value scalar type checking
    with pytest.raises(TypeError, match='start_value must be a scalar'):
        TrackerMHW(None, 0, 120, 5, 2, 4, [5, 4], 2)
    # n_up_stop integer check
    with pytest.raises(TypeError, match='n_up_stop must be an integer'):
        TrackerMHW(None, 0, 120, 5, 2, 4, 40, 1.5)
    # x_min integer or float check
    with pytest.raises(TypeError, match='x_min must be a float or integer'):
        TrackerMHW(None, '5', 120, 5, 2, 4, 40, 2)
    # x_max integer or float check
    with pytest.raises(TypeError, match='x_max must be a float or integer'):
        TrackerMHW(None, 0, '90', 5, 2, 4, 40, 2)
    # start_value is a multiple of base_step
    with pytest.raises(ValueError,
                       match='start_value must be a multiple of base_step'):
        TrackerMHW(None, 0, 120, 5, 2, 4, 41, 2)
    # x_min factor check
    with pytest.raises(ValueError,
                       match='x_min must be a multiple of base_step'):
        TrackerMHW(None, 2, 120, 5, 2, 4, 40, 2)
    # x_max factor check
    with pytest.raises(ValueError,
                       match='x_max must be a multiple of base_step'):
        TrackerMHW(None, 0, 93, 5, 2, 4, 40, 2)

    tr = TrackerMHW(None, 0, 120, 5, 2, 4, 10, 2)
    responses = [True, True, True, True, True]
    with pytest.warns(UserWarning, match='''Tracker {} exceeded x_min or x_max
                      bounds {} times.'''):
        for r in responses:
            tr.respond(r)

    tr = TrackerMHW(None, 0, 120, 5, 2, 4, 40, 2)
    responses = [False, False, False, False, False]
    with pytest.warns(UserWarning, match='exceeded x_min or x_max bounds'):
        for r in responses:
            tr.respond(r)
    assert(not tr.check_valid(3))

    tr = TrackerMHW(None, 0, 120, 5, 2, 4, 40, 2)
    responses = [False, False, False, False, True, False, False, True]
    with pytest.warns(UserWarning, match='exceeded x_min or x_max bounds'):
        for r in responses:
            tr.respond(r)
