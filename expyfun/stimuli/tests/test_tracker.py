import numpy as np
from expyfun.stimuli import TrackerUD, TrackerBinom, TrackerDealer
from expyfun import ExperimentController
from nose.tools import assert_raises
import matplotlib.pyplot as plt


def callback(event_type, value=None, timestamp=None):
    print(event_type, value, timestamp)


std_args = ['test']
std_kwargs = dict(output_dir=None, full_screen=False, window_size=(1, 1),
                  participant='foo', session='01', stim_db=0.0, noise_db=0.0,
                  verbose=True, version='dev')


def test_tracker_ud():
    """Test TrackerUD"""
    tr = TrackerUD(callback, 3, 1, 1, 1, 10, 'trials', 1)
    with ExperimentController(*std_args, **std_kwargs) as ec:
        tr = TrackerUD(ec, 3, 1, 1, 1, 10, 'trials', 1)
    tr = TrackerUD(None, 3, 1, 1, 1, 10, 'trials', 1)
    rand = np.random.RandomState(0)
    while not tr.stopped:
        tr.respond(rand.rand() < tr.x_current)

    tr = TrackerUD(None, 3, 1, 1, 1, 10, 'reversals', 1, x_min=0, x_max=1.1)
    tr.threshold()
    rand = np.random.RandomState(0)
    while not tr.stopped:
        tr.respond(rand.rand() < tr.x_current)
    # test responding after stopped
    assert_raises(RuntimeError, tr.respond, 0)

    # all the properties better work
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
    ax = plt.axes()
    fig, ax, lines = tr.plot(ax)
    plt.close(fig)
    tr.threshold

    # bad callback type
    assert_raises(TypeError, TrackerUD, 'foo', 3, 1, 1, 1, 10, 'trials', 1)

    # test dynamic step size and error conditions
    tr = TrackerUD(None, 3, 1, [1, 0.5], [1, 0.5], 10, 'trials', 1,
                   change_criteria=[0, 2])
    tr.respond(True)
    tr = TrackerUD(None, 3, 1, [1, 0.5], [1, 0.5], 10, 'trials', 1,
                   change_criteria=[0, 2], change_rule='trials')
    tr.respond(True)
    # first element of change_criteria non-zero
    assert_raises(ValueError, TrackerUD, None, 3, 1, [1, 0.5], [1, 0.5], 10,
                  'trials', 1, change_criteria=[1, 2])
    # first element of change_criteria length mistmatch
    assert_raises(ValueError, TrackerUD, None, 3, 1, [1, 0.5], [1, 0.5], 10,
                  'trials', 1, change_criteria=[0])
    # step_size_up length mismatch
    assert_raises(ValueError, TrackerUD, None, 3, 1, [1], [1, 0.5], 10,
                  'trials', 1, change_criteria=[0, 2])
    # step_size_down length mismatch
    assert_raises(ValueError, TrackerUD, None, 3, 1, [1, 0.5], [1], 10,
                  'trials', 1, change_criteria=[0, 2])
    # bad chane_rule
    assert_raises(ValueError, TrackerUD, None, 3, 1, [1, 0.5], [1, 0.5], 10,
                  'trials', 1, change_criteria=[0, 2], change_rule='foo')
    # no change_criteria (i.e. change_criteria=None)
    assert_raises(ValueError, TrackerUD, None, 3, 1, [1, 0.5], [1, 0.5], 10,
                  'trials', 1)


def test_tacker_binom():
    """Test TrackerBinom"""
    tr = TrackerBinom(callback, 0.05, 0.1, 5)
    with ExperimentController(*std_args, **std_kwargs) as ec:
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

    # can't respond before you pick a tracker and get a trial
    assert_raises(RuntimeError, dealer_ud.respond, True)
    rand = np.random.RandomState(0)
    trial = dealer_ud.get_trial()
    while not dealer_ud.stopped:
        trial = dealer_ud.get_trial()
        dealer_ud.respond(rand.rand() < trial[1])
        assert(np.abs(dealer_ud[0].n_reversals -
                      dealer_ud[0].n_reversals) <= 1)

    # can't get a trial after tracker is stopped
    assert_raises(RuntimeError, dealer_ud.get_trial)

    # test array-like indexing
    dealer_ud[0]
    dealer_ud[:]
    dealer_ud[[1, 0, 1]]
    [d for d in dealer_ud]
    dealer_ud.shape
    dealer_ud.trackers
    dealer_ud.history()
    dealer_ud.history(True)

    # bad rand type
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

    # if you're dealing from TrackerBinom, you can't use stop_early feature
    assert_raises(ValueError, TrackerDealer, [2], TrackerBinom,
                  [None, 0.05, 0.5, 50], dict(stop_early=True))
