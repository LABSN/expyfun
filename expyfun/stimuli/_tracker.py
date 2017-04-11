"""Adaptive tracks for psychophysics (individual, or multiple randomly dealt)
"""
# Author: Ross Maddox <ross.maddox@rochester.edu>
#
# License: BSD (3-clause)

import numpy as np
import time
from scipy.stats import binom
import json
import matplotlib.pyplot as plt
from .. import ExperimentController


# =============================================================================
# Set up the logging callback (use write_data_line or do nothing)
# =============================================================================
def _callback_dummy(event_type, value=None, timestamp=None):
    """Take the arguments of write_data_line, but do nothing.
    """
    pass


def _check_callback(callback):
    """Check to see if the callback is of an allowable type.
    """
    if callback is None:
        callback = _callback_dummy
    elif isinstance(callback, ExperimentController):
        callback = callback.write_data_line

    if not callable(callback):
        raise TypeError('callback must be a callable, None, or an instance of '
                        'ExperimentController.')
    return callback


# =============================================================================
# Define the TrackerUD Class
# =============================================================================
class TrackerUD(object):
    """Up-down adaptive tracker

    This class implements a standard up-down adative tracker object. Based on
    how it is configured, it can be used to run a fixed-step m-down n-up
    tracker (staircase), or it can implement a weighted up-down procedure.

    Parameters
    ----------
    callback : callable | ExperimentController | None
        The function that will be used to print the data, usually to the
        experiment .tab file. It should follow the prototype of
        ``ExperimentController.write_data_line``. If an instance of
        ``ExperimentController`` is given, then it will take that object's
        ``write_data_line`` function. If None is given, then it will not write
        the data anywhere.
    up : int
        The number of wrong answers necessary to move the tracker level up.
    down : int
        The number of correct answers necessary to move the tracker level down.
    step_size_up : float | list of float
        The size of the step when the tracker moves up. If float it will stay
        the same. If list of float then it will change when ``change_criteria``
        are encountered. See note below for more specific information on
        dynamic tracker parameters specified with a list.
    step_size_down : float | list of float
        The size of the step when the tracker moves down. If float it will stay
        the same. If list of float then it will change when ``change_criteria``
        are encountered. See note below for more specific information on
        dynamic tracker parameters specified with a list.
    stop_criterion : int
        The number of reversals or trials that will stop the tracker.
    stop_rule : str
        How to determine when the tracker stops. Can be 'reversals' or
        'trials'.
    start_value : float
        The starting level of the tracker.
    change_criteria : list of int | None
        The points along the tracker to change its step sizes. Has an effect
        where ``step_size_up`` and ``step_size_down`` are
        lists. The length of ``change_criteria`` must be the same as the
        length of ``step_size_up`` and ``step_size_down``. See note below for
        more specific usage information. Should be None if static step sizes
        are used.
    change_rule : str
        Whether to change parameters based on 'trials' or 'reversals'.
    x_min : float
        The minimum value that the tracker level (``x``) is allowed to take.
    x_max : float
        The maximum value that the tracker level (``x``) is allowed to take.

    Returns
    -------
    tracker : instance of TrackerUD
        The up-down tracker object.

    Notes
    -----
    It is common to use dynamic parameters in an adaptive tracker. For example:
    the step size is often large for the first couple reversals to get in the
    right general area, and then it is reduced for the remainder of the track.
    This class allows that functionality by defining that parameter with a list
    of values rather than a scalar. The parameter will change to the next value
    in the list whenever a change criterion (number of trials or reversals) is
    encountered. This means that the length of the list defining a dynamic
    parameter must always be the same as that of ``change_criteria``, and the
    first element of change_criteria must always be 0.
    For the example given above:
        ``step_size_up=[1., 0.2]``, ``step_size_down=[1., 0.2]``,
        ``change_criteria=[0, 2]``, ``change_rule='reversals'``
    would change the step sizes from 1 to 0.2 after two reversals.

    If static step sizes are used, both ``step_size_up``
    and ``step_sizedownp`` must be scalars and ``change_criteria`` must be
    None.
    """
    def __init__(self, callback, up, down, step_size_up, step_size_down,
                 stop_criterion, stop_rule, start_value, change_criteria=None,
                 change_rule='reversals', x_min=None, x_max=None):
        self._callback = _check_callback(callback)
        self._up = up
        self._down = down
        self._stop_criterion = stop_criterion
        self._stop_rule = stop_rule
        self._start_value = start_value
        self._x_min = -np.inf if x_min is None else x_min
        self._x_max = np.inf if x_max is None else x_max

        if change_criteria is None:
            change_criteria = [0]
        elif change_criteria[0] != 0:
            raise ValueError('First element of change_criteria must be 0.')
        self._change_criteria = np.asarray(change_criteria)
        if change_rule not in ['trials', 'reversals']:
            raise ValueError("change_rule must be either 'trials' or "
                             "'reversals'")
        self._change_rule = change_rule

        step_size_up = np.atleast_1d(step_size_up)
        if len(step_size_up) != len(change_criteria):
            raise ValueError('If step_size_up is not scalar it must be the '
                             'same length as change_criteria.')
        self._step_size_up = np.asarray(step_size_up, dtype=float)

        step_size_down = np.atleast_1d(step_size_down)
        if len(step_size_down) != len(change_criteria):
            raise ValueError('If step_size_down is not scalar it must be the '
                             'same length as change_criteria.')
        self._step_size_down = np.asarray(step_size_down, dtype=float)

        self._x = np.asarray([start_value], dtype=float)
        if not np.isscalar(start_value):
            raise TypeError('start_value must be a scalar')
        self._x_current = float(start_value)
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
            tracker_type='TrackerUD')))

        self._callback('tracker_%i_init' % self._tracker_id, json.dumps(dict(
            callback=None,
            up=self._up,
            down=self._down,
            step_size_up=[int(s) for s in self._step_size_up],
            step_size_down=[int(s) for s in self._step_size_down],
            stop_criterion=self._stop_criterion,
            stop_rule=self._stop_rule,
            start_value=self._start_value,
            change_criteria=[int(s) for s in self._change_criteria],
            change_rule=self._change_rule,
            x_min=self._x_min,
            x_max=self._x_max)))

    def respond(self, correct):
        """Update the tracker based on the last response.

        Parameters
        ----------
        correct : boolean
            Was the most recent subject response correct?
        """
        if self._stopped:
            raise RuntimeError('Tracker is stopped.')

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
            self._callback('tracker_%i_respond' % self._tracker_id,
                           correct)
        else:
            self._x = self._x[:-1]
            self._callback(
                'tracker_%i_stop' % self._tracker_id, json.dumps(dict(
                    responses=[int(s) for s in self._responses],
                    reversals=[int(s) for s in self._reversals],
                    x=[int(s) for s in self._x])))

    def _stop_here(self):
        if self._stop_rule.lower() == 'reversals':
            self._n_stop = self._n_reversals
        elif self._stop_rule.lower() == 'trials':
            self._n_stop = self._n_trials
        return self._n_stop >= self._stop_criterion

    def _step_index(self):
        if self._change_rule.lower() == 'reversals':
            self._n_change = self._n_reversals
        elif self._stop_rule.lower() == 'trials':
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
        """Has the tracker stopped
        """
        return self._stopped

    @property
    def x(self):
        """The staircase
        """
        return self._x

    @property
    def x_current(self):
        """The current level
        """
        return self._x_current

    @property
    def responses(self):
        """The response history
        """
        return self._responses

    @property
    def n_trials(self):
        """The number of trials so far
        """
        return self._n_trials

    @property
    def n_reversals(self):
        """The number of reversals so far
        """
        return self._n_reversals

    @property
    def reversals(self):
        """The reversal history (0 where there was no reversal)
        """
        return self._reversals

    @property
    def reversal_inds(self):
        """The trial indices which had reversals"""
        return np.where(self._reversals)[0]

    # =========================================================================
    # Display functions
    # =========================================================================
    def plot(self, ax=None):
        """Plot the adaptive track.

        Parameters
        ----------
        ax : AxesSubplot | None
            The axes to make the plot on. If ``None`` defaults to current axes.

        Returns
        -------
        fig : Figure
            The figure handle.
        ax : AxesSubplot
            The axes handle.
        lines : list of Line2D
            The handles to the staircase line and the reversal dots.
        """
        if ax is None:
            fig, ax = plt.subplots(1)
        else:
            fig = ax.figure

        line = ax.plot(1 + np.arange(self._n_trials), self._x, 'k.-')
        dots = ax.plot(1 + np.where(self._reversals > 0)[0],
                       self._x[self._reversals > 0], 'ro')
        ax.set(xlabel='Trial number', ylabel='Level')
        return fig, ax, line + dots

    def plot_thresh(self, n_skip=2, ax=None):
        """Plot a line showing the threshold.

        Parameters
        ----------
            n_skip : int
                See documentation for ``TrackerUD.threshold``.
            ax : Axes
                The handle to the axes object. If None, the current axes will
                be used.

        Returns
        -------
        line : list Line2D
            The handle to the threshold line, as returned from ``plt.plot``.
        """
        if ax is None:
            ax = plt.gca()
        h = ax.plot([1, self._n_trials], [self.threshold(n_skip)] * 2,
                    '--', color='gray')
        return h

    def threshold(self, n_skip=2):
        """Compute the track's threshold.

        Parameters
        ----------
        n_skip : int
            The number of reversals to skip at the beginning when computing
            the threshold.

        Returns
        -------
        threshold : float
            The handle to the threshold line.

        Notes
        -----
        The threshold is computed as the average of the up reversals and the
        average of the down reversals. In this way if there's an unequal number
        of them the assymetry won't bias the threshold estimate.

        This can also be used before the track is stopped if the experimenter
        wishes.
        """
        rev_inds = self.reversal_inds[n_skip:]
        if len(rev_inds) < 1:
            return np.nan
        else:
            return (np.mean(self._x[rev_inds[0::2]]) +
                    np.mean(self._x[rev_inds[1::2]])) / 2


# =============================================================================
# Define the TrackerBinom Class
# =============================================================================
class TrackerBinom(object):
    """Binomial hypothesis testing tracker

    This class implements a tracker that runs a test at each trial with the
    null hypothesis that the stimulus presented has no effect on the subject's
    response. This would happen in the case where the subject couldn't hear a
    stimulus, they were not able to do a task, or they did not understand the
    task. This function's main use is training: a subject may move on to the
    experiment when the null hypothesis that they aren't doing the task has
    been rejected.

    Parameters
    ----------
    callback : callable | ExperimentController | None
        The function that will be used to print the data, usually to the
        experiment .tab file. It should follow the prototype of
        ``ExperimentController.write_data_line``. If an instance of
        ``ExperimentController`` is given, then it will take that object's
        ``write_data_line`` function. If ``None`` is given, then it will not
        write the data anywhere.
    alpha : float
        The p-value which is considered significant. Must be between 0 and 1.
        Note that if ``stop_early`` is ``True`` there is the potential for
        multiple comparisons issues and a more stringent ``alpha`` should be
        considered.
    chance : float
        The chance level of the task being performed. Must be between 0 and 1.
    max_trials : int
        The maximum number of trials to run before stopping the track without
        reaching ``alpha``.
    min_trials : int
        The minimum number of trials to run before allowing the track to stop
        on reaching ``alpha``. Has no effect if ``stop_early`` is ``False``.
    stop_early : boolean
        Whether to stop the adaptive track as soon as ``alpha`` is reached and
        at least ``min_trials`` have been presented.
    x_current : float | None
        The level that you want to run the test at. This has no bearing on how
        the track runs, and it will never change, but it is required to be
        here for ``TrackerDealer``.

    Returns
    -------
    tracker : instance of TrackerBinom
        The binomial tracker object.

    Notes
    -----
    The task, unlike with an adaptive tracker, should be done all at one
    stimulus level, usually an easy one. The point of this tracker is to
    confirm that the subject understands the task instructions and is capable
    of following them.
    """
    def __init__(self, callback, alpha, chance, max_trials, min_trials=0,
                 stop_early=True, x_current=None):
        self._callback = _check_callback(callback)
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
        self._callback('tracker_identify', json.dumps(dict(
            tracker_id=self._tracker_id,
            tracker_type='TrackerBinom')))

        self._callback('tracker_%i_init' % self._tracker_id, json.dumps(dict(
            callback=None,
            alpha=self._alpha,
            chance=self._chance,
            max_trials=self._max_trials,
            min_trials=self._min_trials,
            stop_early=self._stop_early,
            x_current=self._x_current)))

    def respond(self, correct):
        """Update the tracker based on the last response.

        Parameters
        ----------
        correct : boolean
            Was the most recent subject response correct?
        """
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
        if self._n_trials == self._max_trials:
            self._stopped = True

        if self._stopped:
            self._callback(
                'tracker_%i_stop' % self._tracker_id, json.dumps(dict(
                    responses=[int(s) for s in self._responses],
                    p_val=self._p_val,
                    success=int(self.success))))
        else:
            self._callback('tracker_%i_respond' % self._tracker_id, correct)

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
        """The number of incorrect trials so far
        """
        return self._n_wrong

    @property
    def n_correct(self):
        """The number of correct trials so far
        """
        return self._n_correct

    @property
    def pc(self):
        """Proportion correct (0-1, NaN before any responses made)
        """
        return self._pc

    @property
    def responses(self):
        """The response history
        """
        return self._responses

    @property
    def stopped(self):
        """Is the tracker stopped
        """
        return self._stopped

    @property
    def success(self):
        """Has the p-value reached significance
        """
        return self._p_val <= self._alpha

    @property
    def x_current(self):
        """Included only for compatibility with TrackerDealer
        """
        return self._x_current

    @property
    def x(self):
        """Included only for compatibility with TrackerDealer
        """
        return np.array([self._x_current for _ in range(self._n_trials)])

    @property
    def stop_rule(self):
        return 'trials'


# =============================================================================
# Define a container for interleaving several tracks simultaneously
# =============================================================================
# TODO: Make it so you can add a list of values for each dimension (such as the
# phase in a BMLD task) and have it return that

# TODO: eventually, make a BaseTracker class so that TrackerDealer can make
# sure it has the methods / properties it needs
class TrackerDealer(object):
    """Class for selecting and pacing independent simultaneous trackers

    Parameters
    ----------
    shape : list-like
        The dimensions of the tracker container.
    tracker_class : class
        The class for the tracker you want to run, e.g. ``TrackerUD``--not an
        instance.
    tracker_args : list-like
        The arguments used to instantiate each of the trackers.
    tracker_kwargs : dict
        The keyword arguments used to instantiate each of the trackers.
    max_lag : int
        The number of reversals or trials by which the leading tracker may lead
        the lagging one. Whether this uses trials or reversals depends on the
        ``stop_rule`` of the tracker
    rand : numpy.random.RandomState | None
        The random process used to deal the trackers. If None, the process is
        seeded based on the current time.

    Returns
    -------
    dealer : instance of TrackerDealer
        The tracker dealer object.

    Notes
    -----
    The trackers can be accessed like a numpy array, e.g. ``dealer[0, 1, :]``.

    If dealing from TrackerBinom objects (which is probably not a good idea),
    ``stop_early`` must be ``False`` or else they cannot be ensured to keep
    pace.
    """
    def __init__(self, shape, tracker_class, tracker_args, tracker_kwargs,
                 max_lag=1, rand=None):
        # dim will only be used for user output. Will be stored as 0-d
        if np.isscalar(shape):
            shape = [shape]
        self._shape = tuple(shape)
        self._n = np.prod(self._shape)
        self._trackers = [tracker_class(*tracker_args, **tracker_kwargs)
                          for _ in range(self._n)]
        self._pace_rule = self._trackers[0].stop_rule
        self._max_lag = max_lag
        if rand is None:
            self._seed = int(time.time())
            self._rand = np.random.RandomState(self._seed)
        else:
            if not isinstance(rand, np.random.RandomState):
                raise TypeError('rand must be of type '
                                'numpy.random.RandomState')
            self._rand = rand
            self._seed = None
        self._trial_complete = True
        self._tracker_history = np.array([], dtype=int)
        self._response_history = np.array([], dtype=int)
        self._x_history = np.array([], dtype=float)
        if isinstance(self._trackers[0], TrackerBinom):
            if self._trackers[0].stop_early:
                raise ValueError('stop_early must be False to deal trials '
                                 'from a TrackerBinom object')

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

    def __next__(self):  # for py3k compatibility
        return self.next()

    def _pick(self):
        """Decide which tracker from which to draw a trial
        """
        if self.stopped:
            raise RuntimeError('All trackers have stopped.')
        active = np.where(np.invert([t.stopped for t in self._trackers]))[0]

        if self._pace_rule == 'reversals':
            pace = np.asarray([t.n_reversals for t in self._trackers])
        elif self._pace_rule == 'trials':
            pace = np.asarray([t.n_trials for t in self._trackers])
        pace = pace[active]
        lag = pace.max() - pace
        lag_max = lag.max()

        if lag_max > self._max_lag:
            # This should never happen, but handle it if it does
            inds = active[lag == lag_max]
        elif lag_max > 0:
            inds = active[lag > 0]
        else:
            inds = active
        return inds[self._rand.randint(len(inds))]

    def get_trial(self):
        """Selects the tracker from which the next trial should be run

        Returns
        -------
        subscripts : list-like
            The position of the selected tracker.
        x_current : float
            The level of the selected tracker.
        """
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
        """Update the current tracker based on the last response

        Parameters
        ----------
        correct : boolean
            Was the most recent subject response correct?

        Notes
        -----
        ``get_trial`` must be run before ``respond`` can be called.
        """
        if self._trial_complete:
            raise RuntimeError('You must get a trial before you can respond.')
        self._trackers[self._current_tracker].respond(correct)
        self._trial_complete = True
        self._response_history = np.append(self._response_history, correct)

    def history(self, include_skips=False):
        """The history of the dealt trials and the responses

        Inputs
        ------
            include_skips : boolean
                Whether or not to include trials where a tracker was dealt but
                no response was made.

        Returns
        -------
            tracker_history : list of int
                The index of which tracker was dealt on each trial. Note that
                the ints in this list correspond the the raveled index.
            x_history : list of float
                The level of the dealt tracker on each trial.
            response_history : list of boolean
                The response history (i.e., correct or incorrect)
        """
        if include_skips:
            return (self._tracker_history, self._x_history,
                    self._response_history)
        else:
            inds = np.invert(np.isnan(self._response_history))
            return (self._tracker_history[inds], self._x_history[inds],
                    self._response_history[inds].astype(bool))

    @property
    def shape(self):
        return self._shape

    @property
    def stopped(self):
        """Are all the trackers stopped
        """
        return np.all([t.stopped for t in self._trackers])

    @property
    def trackers(self):
        """All of the tracker objects in the container
        """
        return np.reshape(self._trackers, self.shape)
