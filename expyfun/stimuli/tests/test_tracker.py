import numpy as np
import matplotlib
matplotlib.use('Agg')  # noqa
from expyfun.stimuli import TrackerUD, TrackerBinom, TrackerDealer
from expyfun import ExperimentController
from nose.tools import assert_raises
from expyfun._utils import _hide_window, requires_opengl21


def callback(event_type, value=None, timestamp=None):
    print(event_type, value, timestamp)


std_kwargs = dict(output_dir=None, full_screen=False, window_size=(1, 1),
                  participant='foo', session='01', stim_db=0.0, noise_db=0.0,
                  trigger_controller='dummy', response_device='keyboard',
                  audio_controller='pyglet',
                  verbose=True, version='dev')


@_hide_window
@requires_opengl21
def test_tracker_ud():
    """Test TrackerUD"""
    import matplotlib.pyplot as plt
    tr = TrackerUD(callback, 3, 1, 1, 1, 10, 'trials', 1)
    with ExperimentController('test', **std_kwargs) as ec:
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

    # start_value scalar type checking
    assert_raises(TypeError, TrackerUD, None, 3, 1, [1, 0.5], [1, 0.5], 10,
                  'trials', [9, 5], change_criteria=[0, 2])
    assert_raises(TypeError, TrackerUD, None, 3, 1, [1, 0.5], [1, 0.5], 10,
                  'trials', None, change_criteria=[0, 2])


@_hide_window
@requires_opengl21
def test_tracker_binom():
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
    trackers = [[TrackerUD(None, 1, 1, 0.06, 0.02, 20, 'reversals', 1)
                for _ in range(2)] for _ in range(3)]
    dealer_ud = TrackerDealer(callback, trackers)
    
    # can't respond to a trial twice
    dealer_ud.next()
    dealer_ud.respond(True)
    assert_raises(RuntimeError, dealer_ud.respond, True)
    
    dealer_ud = TrackerDealer(callback, np.array(trackers))

    # can't respond before you pick a tracker and get a trial
    assert_raises(RuntimeError, dealer_ud.respond, True)
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
    trackers = [TrackerUD(None, 1, 1, 0.06, 0.02, 20, 'reversals', 1)
                for _ in range(2)]
    assert_raises(TypeError, TrackerDealer, trackers, rand=1)

    # test TrackerDealer with TrackerBinom
    trackers = [TrackerBinom(None, 0.05, 0.5, 50, stop_early=False, 
                             x_current=3) for _ in range(2)]
    dealer_binom = TrackerDealer(callback, trackers)
    for sub, x_current in dealer_binom:
        dealer_binom.respond(True)

    # if you're dealing from TrackerBinom, you can't use stop_early feature
    trackers = [TrackerBinom(None, 0.05, 0.5, 50, stop_early=True)
                for _ in range(2)]
    assert_raises(ValueError, TrackerDealer, callback, trackers)

    # if you're dealing from TrackerBinom, you must include x_current
    trackers = [TrackerBinom(None, 0.05, 0.5, 50, stop_early=True)
                for _ in range(2)]
    assert_raises(ValueError, TrackerDealer, callback, trackers)
