"""Hardware interfaces for key- and button-presses"""

# Authors: Dan McCloy <drmccloy@uw.edu>
#          Eric Larson <larsoner@uw.edu>
#
# License: BSD (3-clause)

import numpy as np
from psychopy import event
from psychopy import clock as psyclock

from ._utils import wait_secs, psylog


class BaseKeyboard(object):
    """Retrieve presses from various devices.

    Public metohds:
        __init__
        listen_presses
        get_presses
        wait_one_press
        wait_for_presses
        check_force_quit

    Requires:
        _get_keyboard_timebase
        _clear_events
        _retrieve_events
    """

    def __init__(self, ec, force_quit_keys):
        self.master_clock = ec._master_clock
        self.log_presses = ec._log_presses
        self.ec_close = ec.close  # needed for check_force_quit
        self.force_quit_keys = force_quit_keys
        self.listen_start = None
        self.time_correction = None
        self.time_correction = self._get_time_correction()

    def _clear_events(self):
        """Clear all events from keyboard buffer.
        """
        raise NotImplementedError

    def _retrieve_events(self, live_keys):
        """Get all events since last call to _clear_events.
        """
        raise NotImplementedError

    def _get_keyboard_timebase(self):
        """Hardware time (seconds) for the keypresses.
        """
        raise NotImplementedError

    def listen_presses(self):
        """Start listening for keypresses.
        """
        self.time_correction = self._get_time_correction()
        self.listen_start = self.master_clock.getTime()
        self._clear_events()

    def get_presses(self, live_keys, timestamp, relative_to):
        """Get the current entire keyboard / button box buffer.
        """
        pressed = []
        if timestamp and relative_to is None:
            if self.listen_start is None:
                raise ValueError('I cannot timestamp: relative_to is None and '
                                 'you have not yet called listen_presses.')
            else:
                relative_to = self.listen_start
        pressed = self._retrieve_events(live_keys)
        return self._correct_presses(pressed, timestamp, relative_to)

    def wait_one_press(self, max_wait, min_wait, live_keys,
                       timestamp, relative_to):
        """Returns only the first button pressed after min_wait.
        """
        relative_to, start_time = self._init_wait_press(max_wait, min_wait,
                                                        live_keys, timestamp,
                                                        relative_to)
        pressed = []
        while (not len(pressed) and
               self.master_clock.getTime() - start_time < max_wait):
            pressed = self._retrieve_events(live_keys)

        # handle non-presses
        if len(pressed):
            pressed = self._correct_presses(pressed, timestamp, relative_to)[0]
        elif timestamp:
            pressed = (None, None)
        else:
            pressed = None
        return pressed

    def wait_for_presses(self, max_wait, min_wait, live_keys,
                         timestamp, relative_to):
        """Returns all button presses between min_wait and max_wait.
        """
        relative_to, start_time = self._init_wait_press(max_wait, min_wait,
                                                        live_keys, timestamp,
                                                        relative_to)

        while (self.master_clock.getTime() - start_time < max_wait):
            pressed = self._retrieve_events(live_keys)
        return self._correct_presses(pressed, timestamp, relative_to)

    def check_force_quit(self, keys=None):
        """Compare key buffer to list of force-quit keys and quit if matched.

        This function always uses the keyboard, so is part of abstraction.
        """
        if keys is None:
            keys = event.getKeys(self.force_quit_keys, timeStamped=False)
        elif type(keys) is str:
            keys = [k for k in [keys] if k in self.force_quit_keys]
        elif type(keys) is list:
            keys = [k for k in keys if k in self.force_quit_keys]
        else:
            raise TypeError('Force quit checking requires a string or list of'
                            ' strings, not a {}.'.format(type(keys)))
        if len(keys):
            self.ec_close()
            raise RuntimeError('Quit key pressed')

    def _get_time_correction(self):
        """Clock correction (seconds) between clocks for hardware and EC.
        """
        other_clock = self._get_keyboard_timebase()
        start_time = self.master_clock.getTime()
        time_correction = start_time - other_clock
        if self.time_correction is not None:
            if not np.allclose(self.time_correction, time_correction,
                               rtol=0, atol=10e-6):
                psylog.warn('Expyfun: drift of > 10 microseconds between '
                            'system clock and experiment master clock.')
            psylog.debug('Expyfun: time correction between system clock and '
                         'experiment master clock is {}. This is a change of '
                         '{} from the previous value.'
                         ''.format(time_correction, time_correction -
                                   self.time_correction))
        return time_correction

    def _correct_presses(self, pressed, timestamp, relative_to):
        """Correct timing of presses and check for quit press"""
        if len(pressed):
            pressed = [(k, s + self.time_correction) for k, s in pressed]
            self.log_presses(pressed)
            keys = [k for k, _ in pressed]
            self.check_force_quit(keys)
            if timestamp:
                pressed = [(k, s - relative_to) for k, s in pressed]
            else:
                pressed = keys
        return pressed

    def _init_wait_press(self, max_wait, min_wait, live_keys, timestamp,
                         relative_to):
        """Actions common to ``wait_one_press`` and ``wait_for_presses``
        """
        if np.isinf(max_wait) and live_keys == []:
            raise ValueError('max_wait cannot be infinite if there are no live'
                             ' keys.')
        if not min_wait < max_wait:
            raise ValueError('min_wait must be less than max_wait')
        start_time = self.master_clock.getTime()
        if timestamp and relative_to is None:
            relative_to = start_time
        wait_secs(min_wait)
        self.check_force_quit()
        self._clear_events()
        return relative_to, start_time


class PsychKeyboard(BaseKeyboard):
    """Retrieve button presses from keyboard.
    """
    def __init__(self, *args, **kwargs):
        BaseKeyboard.__init__(self, *args, **kwargs)
        self._press_buffer = []

    def _clear_events(self):
        event.clearEvents('keyboard')
        self._press_buffer = []

    def _retrieve_events(self, live_keys):
        live_keys = self._add_escape_keys(live_keys)
        self._press_buffer.extend(event.getKeys(live_keys, timeStamped=True))
        return self._press_buffer

    def _get_keyboard_timebase(self):
        """Get psychopy keyboard time reference.

        Notes
        -----
        On Linux, PsychoPy's method for timestamping keypresses from Pyglet
        uses timeit.default_timer, which wraps to time.time and returns seconds
        elapsed since 1970/01/01 at midnight, giving very large numbers. On
        Windows, PsychoPy's timer uses the Windows Query Performance Counter,
        which also returns floats in seconds (apparently measured from when the
        system was last rebooted). Importantly, on either system the units are
        in seconds, and thus can simply be subtracted out.
        """
        return psyclock.getTime()

    def _add_escape_keys(self, live_keys):
        """Helper to add force quit keys to button press listener.
        """
        if live_keys is not None:
            live_keys = [str(x) for x in live_keys]  # accept ints
            if len(self.force_quit_keys):  # should always be a list of strings
                live_keys = live_keys + self.force_quit_keys
        return live_keys
