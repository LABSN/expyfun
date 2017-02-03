# -*- coding: utf-8 -*-
"""Adative tracks for psychophysics (individual or multiple randomly dealt)
"""

import warnings
import numpy as np
import time
from scipy.stats import binom as binom
import json

# =============================================================================
# Set up the logging callback (use write_data_line or do nothing)
# =============================================================================
def callback_dummy(event_type, value=None, timestamp=None):
    """Take the arguments of write_data_line, but do nothing.
    """
    pass


def check_callback(callback):
    """Check see if the callback is of an allowable type.
    """
    if isinstance(callback, 'ExperimentController'):
        callback = callback.write_data_line
    elif callback is None:
        callback = callback_dummy

    if callable(callback) or callback is None:
        return callback
    else:
        raise(TypeError,
              'callback must be a callable, None, or an instance of '
              'ExperimentController.')


# =============================================================================
# Define the TrackerUD Class
# =============================================================================
class TrackerUD(object):
    def __init__(self, callback, up, down, step_size_up, step_size_down,
                 stop_criterion, stop_rule, start_value, change_criteria=None,
                 change_rule='reversals', x_min=None, x_max=None,
                 start_1_down=False):
        self._callback = check_callback(callback)
        self._up = up
        self._down = down
        self._stop_criterion = stop_criterion
        self._stop_rule = stop_rule
        self._start_value = start_value
        self._x_min = x_min
        self._x_max = x_max

        if change_criteria is None:
            change_criteria = [0]
        elif change_criteria[0] != 0:
            raise(ValueError('First element of change points must be 0.'))
        self._change_criteria = np.asarray(change_criteria)
        if change_rule not in ['trials', 'reversals']:
            raise(ValueError, "change_rule must be either 'trials' or "
                  "'reversals'")
        else:
            self._change_rule = change_rule

        if np.isscalar(step_size_up):
            step_size_up = [step_size_up] * len(change_criteria)
        elif len(step_size_up) != len(change_criteria):
            raise(ValueError('If step_size_up is not scalar it must be the '
                             'same length as change_criteria.'))
        self._step_size_up = np.asarray(step_size_up, dtype=float)

        if np.isscalar(step_size_down):
            step_size_down = [step_size_down] * len(change_criteria)
        elif len(step_size_down) != len(change_criteria):
            raise(ValueError('If step_size_down is not scalar it must be the '
                  'same length as change_criteria.'))
        self._step_size_down = np.asarray(step_size_down, dtype=float)

        self._x = np.asarray([start_value], dtype=float)
        self._x_current = start_value
        self._responses = np.asarray([], dtype=bool)
        self._reversals = np.asarray([], dtype=int)
        self._n_up = 0
        self._n_down = 0

        self._direction = 0
        self._n_trials = 0
        self._n_reversals = 0
        self._stopped = False

        # Now write the initialization data out
        self._tracker_id = id(self)
        self._callback('tracker_identify', json.dumps(dict(
            tracker_id=self._tracker_id,
            tracker_type=type(self))))

        self._callback('tracker_%i_init' & self._tracker_id, json.dumps(dict(
            callback=None,
            up=up,
            down=down,
            step_size_up=step_size_up,
            step_size_down=step_size_down,
            stop_criterion=stop_criterion,
            stop_rule=stop_rule,
            start_value=start_value,
            change_criteria=change_criteria,
            change_rule=change_rule,
            x_min=x_min,
            x_max=x_max,
            start_1_down=start_1_down)))

    def respond(self, correct):
        if not self._stopped:
            reversal = False
            self._responses = np.append(self._responses, correct)
            self._n_trials += 1
            step_dir = 0  # 0 no step, 1 up, -1 down

            # Determine if it's a reversal and which way we're going
            if correct:
                self._n_up = 0
                self._n_down += 1
                if self._n_down == self._down:
                    step_dir = -1
                    self._n_down = 0
                    if self._direction > 0:
                        reversal = True
                        self._n_reversals += 1
                    if self._direction >= 0:
                        self._direction = -1
            else:
                self._n_down = 0
                self._n_up += 1
                if self._n_up == self._up:
                    step_dir = 1
                    self._n_up = 0
                    if self._direction < 0:
                        reversal = True
                        self._n_reversals += 1
                    if self._direction <= 0:
                        self._direction = 1
            if reversal:
                self._reversals = np.append(self._reversals, self._n_reversals)
            else:
                self._reversals = np.append(self._reversals, 0)

            # Update the staircase
            if step_dir == 0:
                self._x = np.append(self._x, self._x[-1])
            elif step_dir < 0:
                self._x = np.append(self._x, self._x[-1] -
                                    self._current_step_size_down)
            elif step_dir > 0:
                self._x = np.append(self._x, self._x[-1] +
                                    self._current_step_size_up)

            # Should we stop here?
            self._stopped = self._stop_here()

            if not self._stopped:
                if self._x_min is not None:
                    self._x[-1] = max(self._x_min, self._x[-1])
                if self._x_max is not None:
                    self._x[-1] = min(self._x_max, self._x[-1])

                self._x_current = self._x[-1]
                self._callback('tracker_%i_respond' & self._tracker_id,
                               correct)
            else:
                self._x = self._x[:-1]
                self._callback(
                    'tracker_%i_stop' & self._tracker_id, json.dumps(dict(
                    responses=self._response,
                    reversals=self._reversals,
                    x=self._x)))
        else:
            raise(RuntimeError, 'Tracker is stopped.')

    def _stop_here(self):
        if str.lower(self._stop_rule) == 'reversals':
            self._n_stop = self._n_reversals
        elif str.lower(self._stop_rule) == 'trials':
            self._n_stop = self._n_trials
        return self._n_stop >= self._stop_criterion

    def _step_index(self):
        if str.lower(self._change_rule) == 'reversals':
            self._n_change = self._n_reversals
        elif str.lower(self._stop_rule) == 'trials':
            self._n_change = self._n_trials
        return np.where(self._n_change >= self._change_criteria)[0][-1]

    @property
    def _current_step_size_up(self):
        return self._step_size_up[self._step_index()]

    @property
    def _current_step_size_down(self):
        return self._step_size_down[self._step_index()]

    # =========================================================================
    # Define all the public properties
    # =========================================================================
    @property
    def up(self):
        return self._up

    @property
    def down(self):
        return self._down

    @property
    def step_size_up(self):
        return self._step_size_up

    @property
    def step_size_down(self):
        return self._step_size_down

    @property
    def stop_criterion(self):
        return self._stop_criterion

    @property
    def stop_rule(self):
        return self._stop_rule

    @property
    def start_value(self):
        return self._start_value

    @property
    def x_min(self):
        return self._x_min

    @property
    def x_max(self):
        return self._x_max

    @property
    def stopped(self):
        return self._stopped

    @property
    def x(self):
        return self._x

    @property
    def x_current(self):
        return self._x_current

    @property
    def responses(self):
        return self._responses

    @property
    def n_trials(self):
        return self._n_trials

    @property
    def n_reversals(self):
        return self._n_reversals

    @property
    def reversals(self):
        return self._reversals

    @property
    def reversal_inds(self):
        return np.where(self._reversals)[0]

    # =========================================================================
    # Utility functions
    # =========================================================================
    def _test_prob(self, chance, thresh=None):
        from scipy.stats import norm
        if thresh is None:
            thresh = self._start_value
        slope = np.mean([self._step_size_up.mean(),
                         self._step_size_down.mean()]) ** -1.
        return lambda x: (chance + (1 - chance) *
                          norm.cdf((x - thresh) * slope))

    def _test_iprob(self, chance, thresh=None):
        from scipy.stats import norm
        if thresh is None:
            thresh = self._start_value
        slope = ((self._step_size_up[-1] +
                  self._step_size_down[-1]) / 2) ** -1.
        return lambda x: (thresh +
                          norm.ppf((x - chance) / (1 - chance)) / slope)

    # =========================================================================
    # Display functions
    # =========================================================================
    def plot(self):
        import matplotlib.pyplot as plt
        line = plt.plot(1 + np.arange(self._n_trials), self._x, 'k.-')
        dots = plt.plot(1 + np.where(self._reversals > 0)[0],
                        self._x[self._reversals > 0], 'ro')
        plt.show()
        plt.xlabel('Trial number')
        plt.ylabel('Level')
        return line + dots

    def plot_thresh(self, n_skip=None):
        import matplotlib.pyplot as plt
        plt.plot([1, self._n_trials], [self.threshold(n_skip)] * 2,
                 '--', color='gray')
        plt.show()

    def threshold(self, n_skip=None):
        if n_skip is None:
            if len(self._change_criteria) == 1:
                n_skip = 2
            else:
                n_skip = self._change_criteria[1]
        rev_inds = self.reversal_inds[n_skip:]
        return (np.mean(self._x[rev_inds[0::2]]) +
                np.mean(self._x[rev_inds[1::2]])) / 2


# =============================================================================
# Define the TrackerBinom Class
# =============================================================================
class TrackerBinom(object):
    def __init__(self, callback, alpha, chance, max_trials, min_trials=0,
                 stop_early=True, x_current=None):
        self._callback = check_callback(callback)
        self._alpha = alpha
        self._chance = chance
        self._max_trials = max_trials
        self._min_trials = min_trials  # overrules stop_early
        self._stop_early = stop_early
        self._pval = 1.0
        self._min_p_val = 0.0  # The best they could do given responses so far
        self._max_p_val = 1.0  # The worst they could do given responses so far
        self._n_trials = 0
        self._n_wrong = 0
        self._n_correct = 0
        self._pc = np.nan
        self._responses = np.asarray([], dtype=bool)
        self._stopped = False
        self._x_current = x_current

        # Now write the initialization data out
        self._tracker_id = id(self)
        callback('tracker_identify', json.dumps(dict(
            tracker_id=self._tracker_id,
            tracker_type=type(self))))

        callback('tracker_%i_init' & self._tracker_id, json.dumps(dict(
            callback=callback,
            alpha=alpha,
            chance=chance,
            max_trials=max_trials,
            min_trials=min_trials,
            stop_early=stop_early,
            x_current=x_current)))

    def respond(self, correct):
        self._responses = np.append(self._responses, correct)
        self._n_trials += 1
        if not correct:
            self._n_wrong += 1
        else:
            self._n_correct += 1
        self._pc = float(self._n_correct) / self._n_trials
        self._p_val = binom.cdf(self._n_wrong, self._n_trials,
                                1 - self._chance)
        self._min_p_val = binom.cdf(self._n_wrong, self._max_trials,
                                    1 - self._chance)
        self._max_p_val = binom.cdf(self._n_wrong + (self._max_trials -
                                                     self._n_trials),
                                    self._max_trials, 1 - self._chance)
        if ((self._p_val <= self._alpha) or
                (self._min_p_val >= self._alpha and self._stop_early)):
            if self._n_trials >= self._min_trials:
                self._stopped = True

    # =========================================================================
    # Define all the public properties
    # =========================================================================
    @property
    def alpha(self):
        return self._alpha

    @property
    def chance(self):
        return self._chance

    @property
    def max_trials(self):
        return self._max_trials

    @property
    def stop_early(self):
        return self._stop_early

    @property
    def p_val(self):
        return self._p_val

    @property
    def min_p_val(self):
        return self._min_p_val

    @property
    def max_p_val(self):
        return self._max_p_val

    @property
    def n_trials(self):
        return self._n_trials

    @property
    def n_wrong(self):
        return self._n_wrong

    @property
    def n_correct(self):
        return self._n_correct

    @property
    def pc(self):
        return self._pc

    @property
    def responses(self):
        return self._responses

    @property
    def stopped(self):
        return self._stopped

    @property
    def success(self):
        return self._p_val <= self._alpha

    @property
    def x_current(self):
        return self._x_current

    @property
    def x(self):
        return np.array([self._x_current for _ in range(self._n_trials)])


# =============================================================================
# Define a container for interleaving several tracks simultaneously
# =============================================================================
# TODO: Make it so you can add a list of values for each dimension (such as the
# phase in a BMLD task) and have it return that

# TODO: evemtually, make a BaseTracker class so that Trackers can make sure it
# has the methods / properties it needs
class Trackers(object):
    def __init__(self, tracker, shape, pacer='reversals', slop=1, rand=None):
        from copy import deepcopy
        # dim will only be used for user output. Will be stored as 0-d
        if np.isscalar(shape):
            shape = [shape]
        self._shape = tuple(shape)
        self._n = np.prod(self._shape)
        self._trackers = [deepcopy(tracker) for _ in range(self._n)]
        if pacer not in ['reversals', 'trials']:
            raise(ValueError, "pacer must be either 'reversals' or 'trials'.")
        self._pacer = pacer
        self._slop = slop
        if rand is None:
            self._seed = int(time.time())
            self._rand = np.random.RandomState(seed)
        self._trial_complete = True
        self._tracker_history = np.array([], dtype=int)
        self._response_history = np.array([], dtype=int)
        self._x_history = np.array([], dtype=float)

    def __getitem__(self, key):
        return np.reshape(self._trackers, self._shape)[key]

    def __iter__(self):
        self._index = 0
        return self

    def next(self):
        if self._index == len(self._trackers):
            raise(StopIteration)
        t = self._trackers[self._index]
        self._index += 1
        return t

    def _pick(self):
        if self.stopped:
            raise(RuntimeError('All trackers have stopped.'))
        active = np.where(np.invert([t.stopped for t in self._trackers]))[0]

        if self._pacer == 'reversals':
            pace = np.asarray([t.n_reversals for t in self._trackers])
        elif self._pacer == 'trials':
            pace = np.asarray([t.n_trials for t in self._trackers])
        pace = pace[active]
        lag = pace.max() - pace
        lag_max = lag.max()

        if lag_max > self._slop:
            inds = active[lag == lag_max]
        elif lag_max > 0:
            inds = active[lag > 0]
        else:
            inds = active
        return inds[self._rand.randint(len(inds))]

    def get_trial(self):
        if not self._trial_complete:
            # Chose a new tracker before responding, so record non-response
            self._response_history = np.append(self._response_history,
                                                np.nan)
        self._trial_complete = False
        self._current_tracker = self._pick()
        self._tracker_history = np.append(self._tracker_history,
                                           self._current_tracker)
        ss = np.unravel_index(self._current_tracker, self.shape)
        level = self._trackers[self._current_tracker].x_current
        self._x_history = np.append(self._x_history, level)
        return ss, level

    def respond(self, correct):
        if self._trial_complete:
            raise(RuntimeError, 'You must get a trial before you can respond.')
        self._trackers[self._current_tracker].respond(correct)
        self._trial_complete = True
        self._response_history = np.append(self._response_history, correct)

    @property
    def shape(self):
        return self._shape

    @property
    def stopped(self):
        return np.all([t.stopped for t in self._trackers])

    @property
    def trackers(self):
        return np.reshape(self._trackers, self.shape)

    def history(self, include_skips=False):
        if include_skips:
            return (self._tracker_history, self._x_history,
                    self._response_history)
        else:
            inds = np.invert(np.isnan(self._response_history))
            return (self._tracker_history[inds], self._x_history[inds],
                    self._response_history[inds].astype(bool))


# =============================================================================
# Junk functions for testing (remove on release)
# =============================================================================
# Test the multi-tracker object
def test_Trackers():
    ud = TrackerUD(1, 3, 1, 1, 10, 'reversals', 0)
    t = Trackers(ud, [3, 2], pacer='trials')
    while not t.stopped:
        [inds, level] = t.get_trial()
        if np.random.rand() < 0.95:
            t.respond(np.random.rand() < 0.794)


# Test some tracks
def test_UD():
    import matplotlib.pyplot as plt
    step_size = [8, 4, 2, 1]
    n_rev = [2, 2, 4]
    change_points = np.cumsum([0] + n_rev)
    n_runs = 1000
    threshes = []
    plt.clf()
    plt.subplot(121)
    for ri in range(n_runs):
        ud = TrackerUD(1, 3, step_size, np.asarray(step_size) / 1., 75,
                       'trials', 0, change_criteria=change_points)
        prob = ud._test_prob(0.5, -ud._test_iprob(0.5, 20)(.794))
        while not ud.stopped:
            ud.respond(np.random.rand() < prob(ud.x_current))
        if ri < 10:
            ud.plot_thresh(4)
            ud.plot()
        threshes += [ud.threshold(4)]

    from scipy.stats import gaussian_kde
    plt.subplot(122)
    kde = gaussian_kde(threshes)
    x = np.linspace(-4 * kde.dataset.std(),
                     4 * kde.dataset.std(), 2e2) + kde.dataset.mean()
    plt.plot(x, kde.evaluate(x), 'k')
    plt.grid()
    plt.xlim(x[0], x[-1])
    plt.tight_layout()

    print('%0.2f +/- %0.2f' % (np.mean(threshes), np.std(threshes)))
    print('Perf at thresh: %0.2f%%' % prob(np.mean(threshes)))
    return ud


# Test the bionomial tracker
def testbinom():
    n_success = 0
    n_runs = 100
    pc = 0.8
    for _ in range(n_runs):
        t = TrackerBinom(0.05, 0.5, 25)
        while not t.stopped:
            t.respond(np.random.rand() < pc)
        n_success += t.success
    print(100. * n_success / n_runs)


#Test the binmial tracker in the Trackers container
def testbinom_Trackers():
    t = Trackers(TrackerBinom(0.05, 0.5, 25), [3, 2], pacer='trials')
    while not t.stopped:
        [inds, level] = t.get_trial()
        t.respond(np.random.rand() < 0.8)
    for tr in t:
        print(tr.p_val),