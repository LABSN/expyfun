"""Hardware interfaces for key- and button-presses"""

# Authors: Dan McCloy <drmccloy@uw.edu>
#          Eric Larson <larsoner@uw.edu>
#
# License: BSD (3-clause)

import numpy as np
from functools import partial
import pyglet

from ._utils import wait_secs, clock


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
        ec._time_correction_fxns['keypress'] = self._get_keyboard_timebase
        self.get_time_corr = partial(ec._get_time_correction, 'keypress')
        self.time_correction = self.get_time_corr()
        self.win = ec._win

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
        self.time_correction = self.get_time_corr()
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
            #keys = getKeys(self.force_quit_keys, timeStamped=False)
            keys = []  # NEED TO FIX HANDLING OF FORCE_QUIT CHECKING TDT!
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
        if not min_wait <= max_wait:
            raise ValueError('min_wait must be less than max_wait')
        start_time = self.master_clock.getTime()
        if timestamp and relative_to is None:
            relative_to = start_time
        wait_secs(min_wait)
        self.check_force_quit()
        self._clear_events()
        return relative_to, start_time


class Keyboard(BaseKeyboard):
    """Retrieve button presses from keyboard.
    """
    def __init__(self, *args, **kwargs):
        BaseKeyboard.__init__(self, *args, **kwargs)
        # init pyglet response handler
        self.win.on_key_press = self._on_pyglet_keypress
        self._pyglet_buffer = []
        self._press_buffer = []

    def _on_pyglet_keypress(self, symbol, modifiers, emulated=False):
        """Handler for on_key_press pyglet events"""
        key_time = clock()
        if emulated:
            this_key = unicode(symbol)
        else:
            this_key = pyglet.window.key.symbol_string(symbol).lower()
            this_key = this_key.lstrip('_').lstrip('NUM_')
        self._pyglet_buffer.append((this_key, key_time))

    def _clear_events(self):
        self.win.dispatch_events()
        self._pyglet_buffer = []
        self._press_buffer = []

    def _retrieve_events(self, live_keys):
        live_keys = self._add_escape_keys(live_keys)
        self.win.dispatch_events()  # pump events on pyglet windows
        targets = []
        for key in self._pyglet_buffer:
            if key[0] in live_keys or live_keys is None:
                targets.append(key)
        self._press_buffer.extend(targets)
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
        return clock()

    def _add_escape_keys(self, live_keys):
        """Helper to add force quit keys to button press listener.
        """
        if live_keys is not None:
            live_keys = [str(x) for x in live_keys]  # accept ints
            if len(self.force_quit_keys):  # should always be a list of strings
                live_keys = live_keys + self.force_quit_keys
        return live_keys


class Mouse(object):
    """Class to track mouse properties and events

    Parameters
    ----------
    win : instance of pyglet Window
        The window the mouse is attached to.
    visible : bool
        Initial mouse visibility.
    """
    def __init__(self, window, visible=False):
        self._visible = visible
        self.win = window
        self.set_visible(visible)

    def set_visible(self, visible):
        """Sets the visibility of the mouse

        Parameters
        ----------
        visible : bool
            If True, make mouse visible.
        """
        self.win.set_mouse_visible(visible)
        self._visible = visible

    @property
    def visible(self):
        """Mouse visibility"""
        return self._visible

    @property
    def pos(self):
        """The current position of the mouse in normalized units"""
        last_pos = np.array([self.win._mouse_x, self.win._mouse_y])
        return (last_pos - self.win.size / 2) / (self.win.size / 2)
