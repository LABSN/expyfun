"""Hardware interfaces for key- and button-presses and mouse clicks"""

# Authors: Dan McCloy <drmccloy@uw.edu>
#          Eric Larson <larsoner@uw.edu>
#          Ross Maddox <rkmaddox@uw.edu>
#
# License: BSD (3-clause)

import numpy as np
from functools import partial

from .visual import (Triangle, Rectangle, Circle, Diamond, ConcentricCircles,
                     FixationDot)
from ._utils import wait_secs, clock, string_types


class Keyboard(object):
    """Retrieve presses from various devices.

    Public metohds:
        __init__
        listen_presses
        get_presses
        wait_one_press
        wait_for_presses
        check_force_quit

    Methods to override by subclasses:
        _get_timebase
        _clear_events
        _retrieve_events
    """

    def __init__(self, ec, force_quit_keys):
        self.master_clock = ec._master_clock
        self.log_presses = ec._log_presses
        self.force_quit_keys = force_quit_keys
        self.listen_start = None
        ec._time_correction_fxns['keypress'] = self._get_timebase
        self.get_time_corr = partial(ec._get_time_correction, 'keypress')
        self.time_correction = self.get_time_corr()
        self.win = ec._win
        # always init pyglet response handler for error (and non-error) keys
        self.win.on_key_press = self._on_pyglet_keypress
        self._keyboard_buffer = []

    ###########################################################################
    # Methods to be overridden by subclasses
    def _clear_events(self):
        self._clear_keyboard_events()

    def _retrieve_events(self, live_keys):
        return self._retrieve_keyboard_events(live_keys)

    def _get_timebase(self):
        """Get keyboard time reference (in seconds)
        """
        return clock()

    def _clear_keyboard_events(self):
        self.win.dispatch_events()
        self._keyboard_buffer = []

    def _retrieve_keyboard_events(self, live_keys):
        # add escape keys
        if live_keys is not None:
            live_keys = [str(x) for x in live_keys]  # accept ints
            live_keys.extend(self.force_quit_keys)
        self.win.dispatch_events()  # pump events on pyglet windows
        targets = []
        for key in self._keyboard_buffer:
            if live_keys is None or key[0] in live_keys:
                targets.append(key)
        return targets

    def _on_pyglet_keypress(self, symbol, modifiers, emulated=False):
        """Handler for on_key_press pyglet events"""
        key_time = clock()
        if emulated:
            this_key = str(symbol)
        else:
            from pyglet.window import key
            this_key = key.symbol_string(symbol).lower()
            this_key = this_key.lstrip('_').lstrip('NUM_')
        self._keyboard_buffer.append((this_key, key_time))

    def listen_presses(self):
        """Start listening for keypresses.
        """
        self.time_correction = self.get_time_corr()
        self.listen_start = self.master_clock()
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
               self.master_clock() - start_time < max_wait):
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
        pressed = []
        while (self.master_clock() - start_time < max_wait):
            pressed = self._retrieve_events(live_keys)
        return self._correct_presses(pressed, timestamp, relative_to)

    def check_force_quit(self, keys=None):
        """Compare key buffer to list of force-quit keys and quit if matched.

        This function always uses the keyboard, so is part of abstraction.
        """
        if keys is None:
            # only grab the force-quit keys
            keys = self._retrieve_keyboard_events([])
        else:
            if isinstance(keys, string_types):
                keys = [keys]
            if isinstance(keys, list):
                keys = [k for k in keys if k in self.force_quit_keys]
            else:
                raise TypeError('Force quit checking requires a string or '
                                ' list of strings, not a {}.'
                                ''.format(type(keys)))
        if len(keys):
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
        start_time = self.master_clock()
        if timestamp and relative_to is None:
            relative_to = start_time
        wait_secs(min_wait)
        self.check_force_quit()
        self._clear_events()
        return relative_to, start_time


class Mouse(object):
    """Class to track mouse properties and events

    Parameters
    ----------
    ec : instance of ``ExperimentController``
        The controller for the current experiment
    visible : bool
        Initial mouse visibility.

    Public metohds:
        __init__
        set_visible
        listen_clicks
        get_clicks
        wait_one_click
        wait_for_clicks

    Methods to override by subclasses:
        _get_timebase
        _clear_events
        _retrieve_events
    """

    def __init__(self, ec, visible=False):
        from pyglet.window import mouse
        self.win = ec._win
        self.set_visible(visible)
        self.master_clock = ec._master_clock
        self.log_clicks = ec._log_clicks
        self.listen_start = None
        ec._time_correction_fxns['mouseclick'] = self._get_timebase
        self.get_time_corr = partial(ec._get_time_correction, 'mouseclick')
        self.time_correction = self.get_time_corr()
        self.win = ec._win
        self._check_force_quit = ec.check_force_quit
        self.win.on_mouse_press = self._on_pyglet_mouse_click
        self._mouse_buffer = []
        self._button_names = {mouse.LEFT: 'left', mouse.MIDDLE: 'middle',
                              mouse.RIGHT: 'right'}
        self._button_ids = {'left': mouse.LEFT, 'middle': mouse.MIDDLE,
                            'right': mouse.RIGHT}
        self._legal_types = (Rectangle, Circle)

    def set_visible(self, visible):
        """Sets the visibility of the mouse

        Parameters
        ----------
        visible : bool
            If True, make mouse visible.
        """
        self.win.set_mouse_visible(visible)
        self.win.set_mouse_platform_visible(visible)  # Pyglet workaround
        self._visible = visible

    @property
    def visible(self):
        """Mouse visibility"""
        return self._visible

    @property
    def pos(self):
        """The current position of the mouse in normalized units"""
        x = (self.win._mouse_x - self.win.width / 2.) / (self.win.width / 2.)
        y = (self.win._mouse_y - self.win.height / 2.) / (self.win.height / 2.)
        return np.array([x, y])

    ###########################################################################
    # Methods to be overridden by subclasses
    def _clear_events(self):
        self._clear_mouse_events()

    def _retrieve_events(self, live_buttons):
        return self._retrieve_mouse_events(live_buttons)

    def _get_timebase(self):
        """Get mouse time reference (in seconds)
        """
        return clock()

    def _clear_mouse_events(self):
        self.win.dispatch_events()
        self._mouse_buffer = []

    def _retrieve_mouse_events(self, live_buttons):
        self.win.dispatch_events()  # pump events on pyglet windows
        targets = []
        for button in self._mouse_buffer:
            if live_buttons is None or button[0] in live_buttons:
                targets.append(button)
        return targets

    def _on_pyglet_mouse_click(self, x, y, button, modifiers):
        """Handler for on_mouse_press pyglet events"""
        button_time = clock()
        this_button = self._button_names[button]
        self._mouse_buffer.append((this_button, x, y, button_time))

    def listen_clicks(self):
        """Start listening for mouse clicks.
        """
        self.time_correction = self.get_time_corr()
        self.listen_start = self.master_clock()
        self._clear_events()

    def get_clicks(self, live_buttons, timestamp, relative_to):
        """Get the current entire mouse buffer.
        """
        clicked = []
        if timestamp and relative_to is None:
            if self.listen_start is None:
                raise ValueError('I cannot timestamp: relative_to is None and '
                                 'you have not yet called listen_clicks.')
            else:
                relative_to = self.listen_start
        clicked = self._retrieve_events(live_buttons)
        return self._correct_clicks(clicked, timestamp, relative_to)

    def wait_one_click(self, max_wait, min_wait, live_buttons,
                       timestamp, relative_to, visible):
        """Returns only the first button clicked after min_wait.
        """
        relative_to, start_time, was_visible = self._init_wait_click(
            max_wait, min_wait, live_buttons, timestamp, relative_to, visible)

        clicked = []
        while (not len(clicked) and
               self.master_clock() - start_time < max_wait):
            clicked = self._retrieve_events(live_buttons)

        # handle non-clicks
        if len(clicked):
            clicked = self._correct_clicks(clicked, timestamp, relative_to)[0]
        elif timestamp:
            clicked = (None, None)
        else:
            clicked = None
        return clicked

    def wait_for_clicks(self, max_wait, min_wait, live_buttons,
                        timestamp, relative_to, visible=None):
        """Returns all clicks between min_wait and max_wait.
        """
        relative_to, start_time, was_visible = self._init_wait_click(
            max_wait, min_wait, live_buttons, timestamp, relative_to, visible)

        clicked = []
        while (self.master_clock() - start_time < max_wait):
            clicked = self._retrieve_events(live_buttons)
        return self._correct_clicks(clicked, timestamp, relative_to)

    def wait_for_click_on(self, objects, max_wait, min_wait,
                          live_buttons, timestamp, relative_to):
        """Waits for a click on one of the supplied window objects
        """
        relative_to, start_time, was_visible = self._init_wait_click(
            max_wait, min_wait, live_buttons, timestamp, relative_to, True)

        index = None
        ci = 0
        while (self.master_clock() - start_time < max_wait and
               index is None):
            clicked = self._retrieve_events(live_buttons)
            while ci < len(clicked) and index is None:  # clicks first
                oi = 0
                while oi < len(objects) and index is None:  # then objects
                    if self._point_in_object(clicked[ci][1:3], objects[oi]):
                        index = oi
                    oi += 1
                ci += 1

        # handle non-clicks
        if index is not None:
            clicked = self._correct_clicks(clicked, timestamp, relative_to)[0]
        elif timestamp:
            clicked = (None, None, None, None)
        else:
            clicked = None

        # Since visibility was forced, set back to what it was before call
        self.set_visible(was_visible)
        return clicked, index

    def _correct_clicks(self, clicked, timestamp, relative_to):
        """Correct timing of clicks"""
        if len(clicked):
            clicked = [(b, x, y, s + self.time_correction) for
                       b, x, y, s in clicked]
            self.log_clicks(clicked)
            buttons = [(b, x, y) for b, x, y, _ in clicked]
            self._check_force_quit()
            if timestamp:
                clicked = [(b, x, y, s - relative_to) for
                           b, x, y, s in clicked]
            else:
                clicked = buttons
        return clicked

    def _init_wait_click(self, max_wait, min_wait, live_buttons, timestamp,
                         relative_to, visible):
        """Actions common to ``wait_one_click`` and ``wait_for_clicks``
        """
        if np.isinf(max_wait) and live_buttons == []:
            raise ValueError('max_wait cannot be infinite if there are no live'
                             ' mouse buttons.')
        if not min_wait <= max_wait:
            raise ValueError('min_wait must be less than max_wait')
        if visible not in [True, False, None]:
            raise ValueError('set_visible must be one of (True, False, None)')
        start_time = self.master_clock()
        if timestamp and relative_to is None:
            relative_to = start_time
        wait_secs(min_wait)
        self._check_force_quit()
        self._clear_events()
        was_visible = self.visible
        if visible is not None:
            self.set_visible(visible)
        return relative_to, start_time, was_visible

    # Define some functions for determining if a click point is in an object
    def _point_in_object(self, pos, obj):
        """Determine if a point is within a visual objec
        """
        if isinstance(obj, (Rectangle, Circle, Diamond, Triangle)):
            return self._point_in_tris(pos, obj)
        elif isinstance(obj, (ConcentricCircles, FixationDot)):
            return np.any([self._point_in_tris(pos, c) for c in obj._circles])

    def _point_in_tris(self, pos, obj):
        """Check to see if a point is in any of the triangles
        """
        found = False
        index = 0
        while index < len(obj._tris) and not found:
            points = obj._tris[index:index + 3]
            tri = np.array([[obj._points[2 * p], obj._points[2 * p + 1]] for
                            p in points])
            found = self._point_in_tri(pos, tri)
            index += 3
        return found

    def _point_in_tri(self, pos, tri):
        """Check to see if a point is in a single triangle
        """
        signs = np.sign([np.cross(tri[np.mod(i + 1, 3)] - tri[i],
                                  pos - tri[i]) for i in range(3)])
        if np.all(signs[1:] == signs[0]):
            return True
        else:
            return False


class CedrusBox(Keyboard):
    """Class for Cedrus response boxes

    Note that experiments with Cedrus boxes are limited to ~4 hours due
    to the data type of their counter (milliseconds since start as integers).
    """
    def __init__(self, ec, force_quit_keys):
        import pyxid
        pyxid.use_response_pad_timer = True
        dev = pyxid.get_xid_devices()[0]
        dev.reset_base_timer()
        assert dev.is_response_device()
        self._dev = dev
        self._keyboard_buffer = []
        super(CedrusBox, self).__init__(ec, force_quit_keys)
        ec._time_correction_maxs['keypress'] = 1e-3  # higher tolerance

    def _get_timebase(self):
        """WARNING: For now this will clear the event queue!"""
        self._retrieve_events(None)
        self._dev.con.read_nonblocking(65536)
        t = self._dev.query_base_timer()
        # This drift correction has been empirically determined, see:
        #  https://github.com/cedrus-opensource/pyxid/issues/2
        #  https://gist.github.com/Eric89GXL/c245574a1eaea65348a3
        t *= 0.00100064206973
        return t

    def _clear_events(self):
        self._retrieve_events(None)
        self._keyboard_buffer = []

    def _retrieve_events(self, live_keys):
        # add escape keys
        if live_keys is not None:
            live_keys = [str(x) for x in live_keys]  # accept ints
            live_keys.extend(self.force_quit_keys)
        # pump for events
        self._dev.poll_for_response()
        while self._dev.response_queue_size() > 0:
            key = self._dev.get_next_response()
            if key['pressed']:
                key = [str(key['key'] + 1), key['time'] / 1000.]
                self._keyboard_buffer.append(key)
            self._dev.poll_for_response()
        # check to see if we have matches
        targets = []
        for key in self._keyboard_buffer:
            if live_keys is None or key[0] in live_keys:
                targets.append(key)
        return targets
