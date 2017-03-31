import numpy as np
from expyfun.stimuli import TrackerUD, TrackerBinom, TrackerDealer
from expyfun import ExperimentController
from nose.tools import assert_raises  # , assert_equal, assert_true
import matplotlib.pyplot as plt


def callback(event_type, value=None, timestamp=None):
    print(event_type, value, timestamp)


def test_tracker_ud():
    """Test TrackerUD"""
    tr = TrackerUD(callback, 3, 1, 1, 1, 10, 'trials', 1)
    with ExperimentController('test', output_dir=None, version='dev',
                              participant='', session='') as ec:
        tr = TrackerUD(ec, 3, 1, 1, 1, 10, 'trials', 1)
    tr = TrackerUD(None, 3, 1, 1, 1, 10, 'trials', 1)
    rand = np.random.RandomState(0)
    while not tr.stopped:
        tr.respond(rand.rand() < tr.x_current)

    tr = TrackerUD(None, 3, 1, 1, 1, 10, 'reversals', 1, x_min=0, x_max=1.1)
    tr.threshold
    rand = np.random.RandomState(0)
    while not tr.stopped:
        tr.respond(rand.rand() < tr.x_current)
    assert_raises(RuntimeError, tr.respond, 0)
    tr.up
    tr.down
    tr.step_size_up
    tr.step_size_down
    tr.stop_criterion
    tr.stop_rule
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
    tr.threshold

    assert_raises(TypeError, TrackerUD, 'foo', 3, 1, 1, 1, 10, 'trials', 1)

    # test dynamic step size and error conditions
    tr = TrackerUD(None, 3, 1, [1, 0.5], [1, 0.5], 10, 'trials', 1,
                   change_criteria=[0, 2])
    tr.respond(True)
    tr = TrackerUD(None, 3, 1, [1, 0.5], [1, 0.5], 10, 'trials', 1,
                   change_criteria=[0, 2], change_rule='trials')
    tr.respond(True)
    assert_raises(ValueError, TrackerUD, None, 3, 1, [1, 0.5], [1, 0.5], 10,
                  'trials', 1, change_criteria=[1, 2])
    assert_raises(ValueError, TrackerUD, None, 3, 1, [1, 0.5], [1, 0.5], 10,
                  'trials', 1, change_criteria=[0])
    assert_raises(ValueError, TrackerUD, None, 3, 1, [1], [1, 0.5], 10,
                  'trials', 1, change_criteria=[0, 2])
    assert_raises(ValueError, TrackerUD, None, 3, 1, [1, 0.5], [1], 10,
                  'trials', 1, change_criteria=[0, 2])
    assert_raises(ValueError, TrackerUD, None, 3, 1, [1, 0.5], [1, 0.5], 10,
                  'trials', 1, change_criteria=[0, 2], change_rule='foo')
    assert_raises(ValueError, TrackerUD, None, 3, 1, [1, 0.5], [1, 0.5], 10,
                  'trials', 1)


def test_tacker_binom():
    """Test TrackerBinom"""
    tr = TrackerBinom(callback, 0.05, 0.1, 5)
    with ExperimentController('test', output_dir=None, version='dev',
                              participant='', session='') as ec:
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
    """Test TrackerDealer"""
    # test TrackerDealer with TrackerUD
    dealer_ud = TrackerDealer(
        [2], TrackerUD, [None, 1, 1, 0.06, 0.02, 20, 'reversals', 1], {})
    dealer_ud = TrackerDealer(
        2, TrackerUD, [None, 1, 1, 0.06, 0.02, 20, 'reversals', 1], {},
        rand=np.random.RandomState(0))

    assert_raises(RuntimeError, dealer_ud.respond, True)
    rand = np.random.RandomState(0)
    trial = dealer_ud.get_trial()
    while not dealer_ud.stopped:
        trial = dealer_ud.get_trial()
        dealer_ud.respond(rand.rand() < trial[1])
        assert(np.abs(dealer_ud[0].n_reversals -
                      dealer_ud[0].n_reversals) <= 1)
    assert_raises(RuntimeError, dealer_ud.get_trial)
    dealer_ud[0]
    dealer_ud[:]
    dealer_ud[[1, 0, 1]]
    [d for d in dealer_ud]
    dealer_ud.shape
    dealer_ud.trackers
    dealer_ud.history()
    dealer_ud.history(True)

    assert_raises(TypeError, TrackerDealer, [2], TrackerUD,
                  [None, 1, 1, 0.06, 0.02, 20, 'reversals', 1], {}, rand=1)

    # test TrackerDealer with TrackerBinom
    dealer_binom = TrackerDealer([2], TrackerBinom,
                                 [None, 0.05, 0.5, 50],
                                 dict(stop_early=False))
    rand = np.random.RandomState(0)
    while not dealer_binom.stopped:
        trial = dealer_binom.get_trial()
        dealer_binom.respond(np.random.rand() < trial[1])

    assert_raises(ValueError, TrackerDealer, [2], TrackerBinom,
                  [None, 0.05, 0.5, 50], dict(stop_early=True))
