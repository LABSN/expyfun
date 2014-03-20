"""Tools for controlling experiment execution"""

# Authors: Dan McCloy <drmccloy@uw.edu>
#          Eric Larson <larsoner@uw.edu>
#
# License: BSD (3-clause)

import numpy as np
import os
import warnings
from os import path as op
from functools import partial
from scipy.signal import resample
import traceback as tb
import pyglet
from pyglet import gl

from ._utils import (get_config, verbose_dec, _check_pyglet_version, wait_secs,
                     running_rms, _sanitize, logger, ZeroClock, date_str,
                     _check_units, set_log_file, flush_logger,
                     string_types, input)
from ._tdt_controller import TDTController
from ._trigger_controllers import ParallelTrigger
from ._sound_controllers import PygletSoundController
from ._input_controllers import Keyboard, Mouse
from .visual import Text, Rectangle


class ExperimentController(object):
    """Interface for hardware control (audio, buttonbox, eye tracker, etc.)

    Parameters
    ----------
    exp_name : str
        Name of the experiment.
    audio_controller : str | dict | None
        If audio_controller is None, the type will be read from the system
        configuration file. If a string, can be 'pyglet' or 'tdt', and the
        remaining audio parameters will be read from the machine configuration
        file. If a dict, must include a key 'TYPE' that is either 'pyglet'
        or 'tdt'; the dict can contain other parameters specific to the TDT
        (see documentation for expyfun.TDTController).
    response_device : str | None
        Can only be 'keyboard' currently.  If None, the type will be read
        from the machine configuration file.
    stim_rms : float
        The RMS amplitude that the stimuli were generated at (strongly
        recommended to be 0.01).
    stim_fs : int | float
        The sampling frequency that the stimuli were generated with (samples
        per second).
    stim_db : float
        The desired dB SPL at which to play the stimuli.
    noise_db : float
        The desired dB SPL at which to play the dichotic noise.
    output_dir : str | None
        An absolute or relative path to a directory in which raw experiment
        data will be stored. If output_folder does not exist, it will be
        created. If None, no output data or logs will be saved
        (ONLY FOR TESTING!).
    window_size : list | array | None
        Window size to use. If list or array, it must have two elements.
        If None, the default will be read from the system config,
        falling back to [1920, 1080] if no system config is found.
    screen_num : int | None
        Screen to use. If None, the default will be read from the system
        config, falling back to 0 if no system config is found.
    full_screen : bool
        Should the experiment window be fullscreen?
    force_quit : list
        Keyboard key(s) to utilize as an experiment force-quit button. Can be
        a zero-element list for no force quit support. If None, defaults to
        ``['lctrl', 'rctrl']``.  Using ['escape'] is not recommended due to
        default handling of 'escape' in pyglet.
    participant : str | None
        If ``None``, a GUI will be used to acquire this information.
    session : str | None
        If ``None``, a GUI will be used to acquire this information.
    trigger_controller : str | None
        If ``None``, the type will be read from the system configuration file.
        If a string, must be 'dummy', 'parallel', or 'tdt'. Note that by
        default the mode is 'dummy', since setting up the parallel port
        can be a pain. Can also be a dict with entries 'type' ('parallel')
        and 'address' (e.g., '/dev/parport0').
    verbose : bool, str, int, or None
        If not None, override default verbose level (see expyfun.verbose).
    check_rms : str | None
        Method to use in checking stimulus RMS to ensure appropriate levels.
        Possible values are ``None``, ``wholefile``, and ``windowed`` (the
        default); see ``set_rms_checking`` for details.
    suppress_resamp : bool
        If ``True``, will suppress resampling of stimuli to the sampling
        frequency of the sound output device.

    Returns
    -------
    exp_controller : instance of ExperimentController
        The experiment control interface.

    Notes
    -----
    When debugging, it's useful to use the flush_logs() method to get
    information (based on the level of verbosity) printed to the console.
    """

    @verbose_dec
    def __init__(self, exp_name, audio_controller=None, response_device=None,
                 stim_rms=0.01, stim_fs=44100, stim_db=65, noise_db=45,
                 output_dir='rawData', window_size=None, screen_num=None,
                 full_screen=True, force_quit=None, participant=None,
                 monitor=None, trigger_controller=None, session=None,
                 verbose=None, check_rms='windowed', suppress_resamp=False):
        # initialize some values
        self._stim_fs = stim_fs
        self._stim_rms = stim_rms
        self._stim_db = stim_db
        self._noise_db = noise_db
        self._stim_scaler = None
        self._suppress_resamp = suppress_resamp
        # placeholder for extra actions to do on flip-and-play
        self._on_every_flip = []
        self._on_next_flip = []
        self._on_trial_ok = []
        # placeholder for extra actions to run on close
        self._extra_cleanup_fun = []
        self._id_call_dict = dict(ec_id=self._stamp_ec_id)
        self._ac = None
        self._data_file = None
        self._clock = ZeroClock()
        self._master_clock = self._clock.get_time

        # put anything that could fail in this block to ensure proper cleanup!
        try:
            self.set_rms_checking(check_rms)
            # Check Pyglet version for safety
            _check_pyglet_version(raise_error=True)
            # assure proper formatting for force-quit keys
            if force_quit is None:
                force_quit = ['lctrl', 'rctrl']
            elif isinstance(force_quit, (int, string_types)):
                force_quit = [str(force_quit)]
            if 'escape' in force_quit:
                logger.warn('Expyfun: using "escape" as a force-quit key is '
                            'not recommended because it has special status in '
                            'pyglet.')

            # set up timing
            # Use ZeroClock, which uses the "clock" fn but starts at zero
            self._time_corrections = dict()
            self._time_correction_fxns = dict()

            # dictionary for experiment metadata
            self._exp_info = {'participant': participant, 'session': session,
                              'exp_name': exp_name, 'date': date_str()}

            # session start dialog, if necessary
            fixed_list = ['exp_name', 'date']  # things not editable in GUI
            for key, value in self._exp_info.iteritems():
                if key not in fixed_list and value is not None:
                    if not isinstance(value, string_types):
                        raise TypeError('{} must be string or None'
                                        ''.format(value))
                    fixed_list.append(key)

            if len(fixed_list) < len(self._exp_info):
                _get_items(self._exp_info, fixed=fixed_list, title=exp_name)

            #
            # initialize log file
            #
            if output_dir is not None:
                output_dir = op.abspath(output_dir)
                if not op.isdir(output_dir):
                    os.mkdir(output_dir)
                basename = op.join(output_dir, '{}_{}'
                                   ''.format(self._exp_info['participant'],
                                             self._exp_info['date']))
                self._log_file = basename + '.log'
                set_log_file(self._log_file)
                # initialize data file
                self._data_file = open(basename + '.tab', 'a')
                self._extra_cleanup_fun.append(self._data_file.close)
                self._data_file.write('# ' + str(self._exp_info) + '\n')
                self.write_data_line('event', 'value', 'timestamp')
            else:
                set_log_file(None)
                self._data_file = None

            #
            # set up monitor
            #
            if screen_num is None:
                screen_num = int(get_config('SCREEN_NUM', '0'))
            if monitor is None:
                mon_size = pyglet.window.get_platform().get_default_display()
                mon_size = mon_size.get_screens()[screen_num]
                mon_size = [mon_size.width, mon_size.height]
                mon_size = ','.join([str(d) for d in mon_size])
                monitor = dict()
                width = float(get_config('SCREEN_WIDTH', '51.0'))
                dist = float(get_config('SCREEN_DISTANCE', '48.0'))
                monitor['SCREEN_WIDTH'] = width
                monitor['SCREEN_DISTANCE'] = dist
                mon_size = get_config('SCREEN_SIZE_PIX', mon_size).split(',')
                mon_size = [int(p) for p in mon_size]
                monitor['SCREEN_SIZE_PIX'] = mon_size
            if not isinstance(monitor, dict):
                raise TypeError('monitor must be a dict')
            req_mon_keys = ['SCREEN_WIDTH', 'SCREEN_DISTANCE',
                            'SCREEN_SIZE_PIX']
            if not all([key in monitor for key in req_mon_keys]):
                raise KeyError('monitor must have keys {0}'
                               ''.format(req_mon_keys))
            mon_size = monitor['SCREEN_SIZE_PIX']
            monitor['SCREEN_DPI'] = (monitor['SCREEN_SIZE_PIX'][0] /
                                     (monitor['SCREEN_WIDTH'] * 0.393701))
            monitor['SCREEN_HEIGHT'] = (monitor['SCREEN_WIDTH']
                                        / float(monitor['SCREEN_SIZE_PIX'][0])
                                        * float(monitor['SCREEN_SIZE_PIX'][1]))
            self._monitor = monitor

            #
            # parse audio controller
            #
            if audio_controller is None:
                audio_controller = {'TYPE': get_config('AUDIO_CONTROLLER',
                                                       'pyglet')}
            elif isinstance(audio_controller, string_types):
                if audio_controller.lower() in ['pyglet', 'tdt']:
                    audio_controller = {'TYPE': audio_controller.lower()}
                else:
                    raise ValueError('audio_controller must be \'pyglet\' or '
                                     '\'tdt\' (or a dict including \'TYPE\':'
                                     ' \'pyo\' or \'TYPE\': \'tdt\').')
            elif not isinstance(audio_controller, dict):
                raise TypeError('audio_controller must be a str or dict.')
            self._audio_type = audio_controller['TYPE'].lower()

            #
            # parse response device
            #
            if response_device is None:
                response_device = get_config('RESPONSE_DEVICE', 'keyboard')
            if response_device not in ['keyboard', 'tdt']:
                raise ValueError('response_device must be "keyboard", "tdt", '
                                 'or None')
            self._response_device = response_device

            #
            # Initialize devices
            #

            # Audio (and for TDT, potentially keyboard)
            if self._audio_type == 'tdt':
                logger.info('Expyfun: Setting up TDT')
                self._ac = TDTController(audio_controller)
                self._audio_type = self._ac.model
            elif self._audio_type == 'pyo':
                self._ac = PyoSound(self, self.stim_fs)
            else:
                raise ValueError('audio_controller[\'TYPE\'] must be '
                                 '\'pyo\' or \'tdt\'.')
            self._extra_cleanup_fun.append(self._ac.halt)
            # audio scaling factor; ensure uniform intensity across devices
            self.set_stim_db(self._stim_db)
            self.set_noise_db(self._noise_db)

            if self._fs_mismatch:
                if self._suppress_resamp:
                    msg = ('Expyfun: Mismatch between reported stim sample '
                           'rate ({0}) and device sample rate ({1}). Nothing '
                           'will be done about this because suppress_resamp '
                           'is "True".'.format(self.stim_fs, self.fs))
                    logger.warn(msg)
                else:
                    msg = ('Expyfun: Mismatch between reported stim sample '
                           'rate ({0}) and device sample rate ({1}). '
                           'Experiment Controller will resample for you, but '
                           'this takes a non-trivial amount of processing '
                           'time and may compromise your experimental '
                           'timing and/or cause artifacts.'
                           ''.format(self.stim_fs, self.fs))
                    logger.warn(msg)

            #
            # set up visual window (must be done before keyboard and mouse)
            #
            logger.info('Expyfun: Setting up screen')
            if full_screen:
                window_size = monitor['SCREEN_SIZE_PIX']
            else:
                if window_size is None:
                    window_size = get_config('WINDOW_SIZE',
                                             '800,600').split(',')
                    window_size = [int(w) for w in window_size]
            window_size = np.array(window_size)
            if window_size.ndim != 1 or window_size.size != 2:
                raise ValueError('window_size must be 2-element array-like or '
                                 'None')

            # open window and setup GL config
            self._setup_window(window_size, exp_name, full_screen, screen_num)

            # Keyboard
            if response_device == 'keyboard':
                self._response_handler = Keyboard(self, force_quit)
            if response_device == 'tdt':
                if not isinstance(self._ac, TDTController):
                    raise ValueError('response_device can only be "tdt" if '
                                     'tdt is used for audio')
                self._response_handler = self._ac
                self._ac._add_keyboard_init(self, force_quit)

            #
            # set up trigger controller
            #
            if trigger_controller is None:
                trigger_controller = get_config('TRIGGER_CONTROLLER', 'dummy')
            if isinstance(trigger_controller, string_types):
                trigger_controller = dict(type=trigger_controller)
            logger.info('Expyfun: Initializing {} triggering mode'
                        ''.format(trigger_controller['type']))
            if trigger_controller['type'] == 'tdt':
                if not isinstance(self._ac, TDTController):
                    raise ValueError('trigger_controller can only be "tdt" if '
                                     'tdt is used for audio')
                _ttl_stamp_func = self._ac.stamp_triggers
            elif trigger_controller['type'] in ['parallel', 'dummy']:
                if 'address' not in trigger_controller['type']:
                    addr = get_config('TRIGGER_ADDRESS')
                    trigger_controller['address'] = addr
                out = ParallelTrigger(trigger_controller['type'],
                                      trigger_controller.get('address'))
                _ttl_stamp_func = out.stamp_triggers
                self._extra_cleanup_fun.append(out.close)
            else:
                raise ValueError('trigger_controller type must be '
                                 '"parallel", "dummy", or "tdt", not '
                                 '{0}'.format(trigger_controller['type']))
            self._id_call_dict['ttl_id'] = self._stamp_binary_id
            self._ttl_stamp_func = _ttl_stamp_func

            # other basic components
            self._mouse_handler = Mouse(self._win)

            # finish initialization
            logger.info('Expyfun: Initialization complete')
            logger.exp('Expyfun: Subject: {0}'
                       ''.format(self._exp_info['participant']))
            logger.exp('Expyfun: Session: {0}'
                       ''.format(self._exp_info['session']))
            self._on_trial_ok.append(self.flush)
            self._trial_identified = False
            self._ofp_critical_funs = list()
        except Exception:
            self.close()
            raise

    def __repr__(self):
        """Return a useful string representation of the experiment
        """
        string = ('<ExperimentController ({3}): "{0}" {1} ({2})>'
                  ''.format(self._exp_info['exp_name'],
                            self._exp_info['participant'],
                            self._exp_info['session'],
                            self._audio_type))
        return string

############################### SCREEN METHODS ###############################
    def screen_text(self, text, pos=[0, 0], color='white', font_name='Arial',
                    font_size=24):
        """Show some text on the screen.

        Parameters
        ----------
        text : str
            The text to be rendered.
        pos : list | tuple
            x, y position of the text. In the default units (-1 to 1, with
            positive going up and right) the default is dead center (0, 0).
        h_align, v_align : str
            Horizontal/vertical alignment of the text relative to ``pos``
        units : str
            units for ``pos``.

        Returns
        -------
        Instance of visual.Text
        """
        scr_txt = Text(self, text, pos, color, font_name, font_size)
        scr_txt.draw()
        self.call_on_next_flip(self.write_data_line, 'screen_text', text)
        return scr_txt

    def screen_prompt(self, text, max_wait=np.inf, min_wait=0, live_keys=None,
                      timestamp=False, clear_after=True):
        """Display text and (optionally) wait for user continuation

        Parameters
        ----------
        text : str | list
            The text to display. It will automatically wrap lines.
            If list, the prompts will be displayed sequentially.
        max_wait : float
            The maximum amount of time to wait before returning. Can be np.inf
            to wait until the user responds.
        min_wait : float
            The minimum amount of time to wait before returning. Useful for
            avoiding subjects missing instructions.
        live_keys : list | None
            The acceptable list of buttons or keys to use to advance the trial.
            If None, all buttons / keys will be accepted.  If an empty list,
            the prompt displays until max_wait seconds have passed.
        clear_after : bool
            If True, the screen will be cleared before returning.

        Returns
        -------
        pressed : tuple | str | None
            If ``timestamp==True``, returns a tuple ``(str, float)`` indicating
            the first key pressed and its timestamp (or ``(None, None)`` if no
            acceptable key was pressed between ``min_wait`` and ``max_wait``).
            If ``timestamp==False``, returns a string indicating the first key
            pressed (or ``None`` if no acceptable key was pressed).
        """
        if not isinstance(text, list):
            text = [text]
        if not all([isinstance(t, string_types) for t in text]):
            raise TypeError('text must be a string or list of strings')
        for t in text:
            self.screen_text(t)
            self.flip()
            out = self.wait_one_press(max_wait, min_wait, live_keys,
                                      timestamp)
        if clear_after:
            self.flip()
        return out

    def draw_background_color(self, color='black'):
        """Draw a solid background color

        Parameters
        ----------
        color : matplotlib color
            The background color.

        Returns
        -------
        rect : instance of Rectangle
            The drawn Rectangle object.

        Notes
        -----
        This should be the first object drawn to a buffer, as it will
        cover any previsouly drawn objects.
        """
        # we go a little over here to be safe from round-off errors
        rect = Rectangle(self, pos=[0, 0, 2.1, 2.1], fill_color=color)
        rect.draw()
        return rect

    def flip_and_play(self, start_of_trial=True):
        """Flip screen, play audio, then run any "on-flip" functions.

        Parameters
        ----------
        start_of_trial : bool
            If True, it checks to make sure that the trial ID has been
            stamped appropriately. Set to False only in cases where
            ``flip_and_play`` is to be used mid-trial (should be rare!).

        Returns
        -------
        flip_time : float
            The timestamp of the screen flip.

        Notes
        -----
        Order of operations is: screen flip, audio start, additional functions
        added with ``on_next_flip``, followed by functions added with
        ``on_every_flip``.
        """
        if start_of_trial:
            if not self._trial_identified:
                raise RuntimeError('Trial ID must be stamped before starting '
                                   'the trial')
            self._trial_identified = False
        logger.exp('Expyfun: Flipping screen and playing audio')
        # ensure self._play comes first in list, followed by other critical
        # private functions (e.g., EL stamping), then user functions:
        self._on_next_flip = ([self._play] + self._ofp_critical_funs +
                              self._on_next_flip)
        flip_time = self.flip()
        return flip_time

    def play(self):
        """Start audio playback

        Returns
        -------
        play_time : float
            The timestamp of the audio playback.
        """
        logger.exp('Expyfun: Playing audio')
        # ensure self._play comes first in list:
        self._play()
        play_time = self._clock.get_time()
        return play_time

    def call_on_next_flip(self, function, *args, **kwargs):
        """Add a function to be executed on next flip only.

        Parameters
        ----------
        function : function | None
            The function to call. If ``None``, all the "on every flip"
            functions will be cleared.

        *args
        -----
        Function arguments.

        **kwargs
        --------
        Function keyword arguments.

        Notes
        -----
        See ``flip_and_play`` for order of operations. Can be called multiple
        times to add multiple functions to the queue.
        """
        if function is not None:
            function = partial(function, *args, **kwargs)
            self._on_next_flip.append(function)
        else:
            self._on_next_flip = []

    def call_on_every_flip(self, function, *args, **kwargs):
        """Add a function to be executed on every flip.

        Parameters
        ----------
        function : function | None
            The function to call. If ``None``, all the "on every flip"
            functions will be cleared.

        *args
        -----
        Function arguments.

        **kwargs
        --------
        Function keyword arguments.

        Notes
        -----
        See ``flip_and_play`` for order of operations. Can be called multiple
        times to add multiple functions to the queue.
        """
        if function is not None:
            function = partial(function, *args, **kwargs)
            self._on_every_flip.append(function)
        else:
            self._on_every_flip = []

    def _convert_units(self, verts, fro, to):
        """Convert between different screen units"""
        _check_units(to)
        _check_units(fro)
        verts = np.array(np.atleast_2d(verts), dtype=float)
        if verts.shape[0] != 2:
            raise RuntimeError('verts must have 2 rows')

        if fro == to:
            return verts

        # simplify by using two if neither is in normalized (native) units
        if 'norm' not in [to, fro]:
            # convert to normal
            verts = self._convert_units(verts, fro, 'norm')
            # convert from normal to dest
            verts = self._convert_units(verts, 'norm', to)
            return verts

        # figure out our actual transition, knowing one is 'norm'
        w_pix = self.window_size_pix[0]
        h_pix = self.window_size_pix[1]
        d_cm = self._monitor['SCREEN_DISTANCE']
        w_cm = self._monitor['SCREEN_WIDTH']
        h_cm = self._monitor['SCREEN_HEIGHT']
        w_prop = w_pix / float(self.monitor_size_pix[0])
        h_prop = h_pix / float(self.monitor_size_pix[1])
        if 'pix' in [to, fro]:
            if 'pix' == to:
                # norm to pixels
                x = np.array([[w_pix / 2., 0, w_pix / 2.],
                              [0, h_pix / 2., h_pix / 2.]])
            else:
                # pixels to norm
                x = np.array([[2. / w_pix, 0, -1.],
                              [0, 2. / h_pix, -1.]])
            verts = np.dot(x, np.r_[verts, np.ones((1, verts.shape[1]))])
        elif 'deg' in [to, fro]:
            if 'deg' == to:
                # norm (window) to norm (whole screen), then to deg
                x = np.arctan2(verts[0] * w_prop * (w_cm / 2.), d_cm)
                y = np.arctan2(verts[1] * h_prop * (h_cm / 2.), d_cm)
                verts = np.array([x, y])
                verts *= (180. / np.pi)
            else:
                # deg to norm (whole screen), to norm (window)
                verts *= (np.pi / 180.)
                x = (d_cm * np.tan(verts[0])) / (w_cm / 2.) / w_prop
                y = (d_cm * np.tan(verts[1])) / (h_cm / 2.) / h_prop
                verts = np.array([x, y])
        else:
            raise KeyError('unknown conversion "{}" to "{}"'.format(fro, to))
        return verts

    def screenshot(self):
        """Capture the current displayed buffer

        This method must be called *before* flipping, because it captures
        the back buffer.

        Returns
        -------
        scr : array
            N x M x 3 array of screen pixel colors.
        """
        # next line must be done in order to instantiate image_buffer_manager
        data = pyglet.image.get_buffer_manager().get_color_buffer()
        data = self._win.context.image_buffer_manager.color_buffer.image_data
        data = data.get_data(data.format, data.pitch)
        data = np.fromstring(data, dtype=np.uint8)
        data.shape = (self._win.width, self._win.height, 4)
        data = np.flipud(data)
        return data

    @property
    def on_next_flip_functions(self):
        """Current stack of functions to be called on next flip."""
        return self._on_next_flip

    @property
    def on_every_flip_functions(self):
        """Current stack of functions called on every flip."""
        return self._on_every_flip

    @property
    def window(self):
        """Pyglet visual window handle."""
        return self._win

    @property
    def dpi(self):
        return self._monitor['SCREEN_DPI']

    @property
    def window_size_pix(self):
        return np.array([self._win.width, self._win.height])

    @property
    def monitor_size_pix(self):
        return np.array(self._monitor['SCREEN_SIZE_PIX'])

############################### OPENGL METHODS ###############################
    def _setup_window(self, window_size, exp_name, full_screen, screen_num):
        config = gl.Config(depth_size=8, double_buffer=True,
                           stencil_size=0, stereo=False)
        max_try = 5  # sometimes it fails for unknown reasons
        for ii in range(max_try):
            try:
                win = pyglet.window.Window(width=window_size[0],
                                           height=window_size[1],
                                           caption=exp_name,
                                           fullscreen=full_screen,
                                           config=config,
                                           screen=screen_num,
                                           style='borderless',
                                           visible=False)
            except pyglet.gl.ContextException:
                if ii == max_try - 1:
                    raise
                else:
                    pass
            else:
                break
        if not full_screen:
            x = int(win.screen.width / 2. - win.width / 2.)
            y = int(win.screen.height / 2. - win.height / 2.)
            win.set_location(x, y)
        self._win = win
        # with the context set up, do basic GL initialization
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)  # set the color to clear to
        gl.glClearDepth(1.0)  # clear value for the depth buffer
        # set the viewport size
        gl.glViewport(0, 0, int(self.window_size_pix[0]),
                      int(self.window_size_pix[1]))
        # set the projection matrix
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.gluOrtho2D(-1, 1, -1, 1)
        # set the model matrix
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        # disable depth testing
        gl.glDisable(gl.GL_DEPTH_TEST)
        # enable blending
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        # set color shading (FLAT or SMOOTH)
        gl.glShadeModel(gl.GL_SMOOTH)
        gl.glEnable(gl.GL_POINT_SMOOTH)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        v_ = False if os.getenv('_EXPYFUN_WIN_INVISIBLE') == 'true' else True
        win.set_visible(v_)
        win.dispatch_events()

    def flip(self):
        """Flip screen, then run any "on-flip" functions.

        Returns
        -------
        flip_time : float
            The timestamp of the screen flip.

        Notes
        -----
        Order of operations is: screen flip, audio start, additional functions
        added with ``on_every_flip``, followed by functions added with
        ``on_next_flip``.
        """
        call_list = self._on_next_flip + self._on_every_flip
        self._win.dispatch_events()
        self._win.flip()
        gl.glTranslatef(0.0, 0.0, -5.0)
        gl.glLoadIdentity()
        #waitBlanking
        gl.glBegin(gl.GL_POINTS)
        gl.glColor4f(0.0, 0.0, 0.0, 0.0)  # transparent
        gl.glVertex2i(10, 10)
        gl.glEnd()
        # this waits until everything is called, including last draw
        gl.glFinish()
        flip_time = self._clock.get_time()
        for function in call_list:
            function()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        self.write_data_line('flip', flip_time)
        self._on_next_flip = []
        return flip_time

    def estimate_screen_fs(self, n_rep=10):
        """Estimate screen refresh rate using repeated flip() calls

        Useful for verifying that a system is operating at the proper
        sample rate.

        Parameters
        ----------
        n_rep : int
            Number of flips to use. The higher the number, the more accurate
            the estimate but the more time will be consumed.

        Returns
        -------
        screen_fs : float
            The screen refresh rate.
        """
        n_rep = int(n_rep)
        times = [self.flip() for _ in range(n_rep)]
        return 1. / np.median(np.diff(times[1:]))

############################ KEYPRESS METHODS ############################
    def listen_presses(self):
        """Start listening for keypresses.
        """
        self._response_handler.listen_presses()

    def get_presses(self, live_keys=None, timestamp=True, relative_to=None):
        """Get the entire keyboard / button box buffer.

        Parameters
        ----------
        live_keys : list | None
            List of strings indicating acceptable keys or buttons. Other data
            types are cast as strings, so a list of ints will also work.
            live_keys=None accepts all keypresses.
        timestamp : bool
            Whether the keypress should be timestamped. If True, returns the
            button press time relative to the value given in ``relative_to``.
        relative_to : None | float
            A time relative to which timestamping is done. Ignored if
            timestamp==False.  If ``None``, timestamps are relative to the time
            ``listen_presses`` was last called.
        """
        return self._response_handler.get_presses(live_keys, timestamp,
                                                  relative_to)

    def wait_one_press(self, max_wait=np.inf, min_wait=0.0, live_keys=None,
                       timestamp=True, relative_to=None):
        """Returns only the first button pressed after min_wait.

        Parameters
        ----------
        max_wait : float
            Duration after which control is returned if no key is pressed.
        min_wait : float
            Duration for which to ignore keypresses (force-quit keys will
            still be checked at the end of the wait).
        live_keys : list | None
            List of strings indicating acceptable keys or buttons. Other data
            types are cast as strings, so a list of ints will also work.
            ``live_keys=None`` accepts all keypresses.
        timestamp : bool
            Whether the keypress should be timestamped. If ``True``, returns
            the button press time relative to the value given in
            ``relative_to``.
        relative_to : None | float
            A time relative to which timestamping is done. Ignored if
            ``timestamp==False``.  If ``None``, timestamps are relative to the
            time ``wait_one_press`` was called.

        Returns
        -------
        pressed : tuple | str | None
            If ``timestamp==True``, returns a tuple (str, float) indicating the
            first key pressed and its timestamp (or ``(None, None)`` if no
            acceptable key was pressed between ``min_wait`` and ``max_wait``).
            If ``timestamp==False``, returns a string indicating the first key
            pressed (or ``None`` if no acceptable key was pressed).
        """
        return self._response_handler.wait_one_press(max_wait, min_wait,
                                                     live_keys, timestamp,
                                                     relative_to)

    def wait_for_presses(self, max_wait, min_wait=0.0, live_keys=None,
                         timestamp=True, relative_to=None):
        """Returns all button presses between min_wait and max_wait.

        Parameters
        ----------
        max_wait : float
            Duration after which control is returned.
        min_wait : float
            Duration for which to ignore keypresses (force-quit keys will
            still be checked at the end of the wait).
        live_keys : list | None
            List of strings indicating acceptable keys or buttons. Other data
            types are cast as strings, so a list of ints will also work.
            ``live_keys=None`` accepts all keypresses.
        timestamp : bool
            Whether the keypresses should be timestamped. If ``True``, returns
            the button press time relative to the value given in
            ``relative_to``.
        relative_to : None | float
            A time relative to which timestamping is done. Ignored if
            ``timestamp`` is ``False``.  If ``None``, timestamps are relative
            to the time ``wait_for_presses`` was called.

        Returns
        -------
        presses : list
            If timestamp==False, returns a list of strings indicating which
            keys were pressed. Otherwise, returns a list of tuples
            (str, float) of keys and their timestamps. If no keys are pressed,
            returns [].
        """
        return self._response_handler.wait_for_presses(max_wait, min_wait,
                                                       live_keys, timestamp,
                                                       relative_to)

    def _log_presses(self, pressed):
        """Write key presses to data file.
        """
        # This function will typically be called by self._response_handler
        # after it retrieves some button presses
        for key, stamp in pressed:
            self.write_data_line('keypress', key, stamp)

    def check_force_quit(self):
        """Check to see if any force quit keys were pressed
        """
        self._response_handler.check_force_quit()

############################# MOUSE METHODS ##################################
    def get_mouse_position(self, units='pix'):
        """Mouse position in screen coordinates

        Parameters
        ----------
        units : str
            Units to return.

        Returns
        -------
        position : ndarray
            The mouse position.
        """
        _check_units(units)
        pos = np.array(self._mouse_handler.pos)
        pos = self._convert_units(pos[:, np.newaxis], 'norm', units)[:, 0]
        return pos

    def toggle_cursor(self, visibility, flip=False):
        """Show or hide the mouse

        Parameters
        ----------
        visibility : bool
            If True, show; if False, hide.
        """
        self._mouse_handler.set_visible(visibility)
        if flip:
            self.flip()

################################ AUDIO METHODS ###############################
    def start_noise(self):
        """Start the background masker noise."""
        self._ac.start_noise()

    def stop_noise(self):
        """Stop the background masker noise."""
        if self._ac is not None:  # check b/c used by __exit__
            self._ac.stop_noise()

    def clear_buffer(self):
        """Clear audio data from the audio buffer."""
        self._ac.clear_buffer()
        logger.exp('Expyfun: Buffer cleared')

    def load_buffer(self, samples):
        """Load audio data into the audio buffer.

        Parameters
        ----------
        samples : np.array
            Audio data as floats scaled to (-1,+1), formatted as an Nx1 or Nx2
            numpy array with dtype float32.
        """
        samples = self._validate_audio(samples) * self._stim_scaler
        logger.exp('Expyfun: Loading {} samples to buffer'
                   ''.format(samples.size))
        self._ac.load_buffer(samples)

    def _play(self):
        """Play the audio buffer.
        """
        logger.debug('Expyfun: playing audio')
        self._ac.play()
        self.write_data_line('play')

    def stop(self):
        """Stop audio buffer playback and reset cursor to beginning of buffer.
        """
        if self._ac is not None:  # need to check b/c used in __exit__
            self._ac.stop()
        self.write_data_line('stop')
        logger.exp('Expyfun: Audio stopped and reset.')

    def set_noise_db(self, new_db):
        """Set the level of the background noise.
        """
        # Noise is always generated at an RMS of 1
        self._ac.set_noise_level(self._update_sound_scaler(new_db, 1.0))
        self._noise_db = new_db

    def set_stim_db(self, new_db):
        """Set the level of the stimuli.
        """
        self._stim_db = new_db
        self._stim_scaler = self._update_sound_scaler(new_db, self._stim_rms)
        # not immediate: new value is applied on the next load_buffer call

    def _update_sound_scaler(self, desired_db, orig_rms):
        """Calcs coefficient ensuring stim ampl equivalence across devices.
        """
        exponent = (-(_get_dev_db(self._audio_type) - desired_db) / 20.0)
        return (10 ** exponent) / float(orig_rms)

    def _validate_audio(self, samples):
        """Converts audio sample data to the required format.

        Parameters
        ----------
        samples : list | array
            The audio samples.  Mono sounds will be converted to stereo.

        Returns
        -------
        samples : numpy.array(dtype='float32')
            The correctly formatted audio samples.
        """
        # check data type
        if type(samples) is list:
            samples = np.asarray(samples, dtype='float32')
        elif samples.dtype != 'float32':
            samples = np.float32(samples)

        # check values
        if np.max(np.abs(samples)) > 1:
            raise ValueError('Sound data exceeds +/- 1.')
            # samples /= np.max(np.abs(samples),axis=0)

        # check dimensionality
        if samples.ndim > 2:
            raise ValueError('Sound data has more than two dimensions.')

        # check shape
        if samples.ndim == 2 and min(samples.shape) > 2:
            raise ValueError('Sound data has more than two channels.')
        elif len(samples.shape) == 2 and samples.shape[0] <= 2:
            samples = samples.T

        # resample if needed
        if self._fs_mismatch and not self._suppress_resamp:
            logger.warn('Expyfun: Resampling {} seconds of audio'
                        ''.format(round(len(samples) / self.stim_fs), 2))
            num_samples = len(samples) * self.fs / float(self.stim_fs)
            samples = resample(samples, int(num_samples), window='boxcar')

        # make stereo if not already
        if samples.ndim == 1:
            samples = np.array((samples, samples)).T
        elif 1 in samples.shape:
            samples = samples.ravel()
            samples = np.array((samples, samples)).T

        # check RMS
        if self._check_rms is not None:
            chans = [samples[:, x] for x in range(samples.shape[1])]
            if self._check_rms == 'wholefile':
                chan_rms = [np.sqrt(np.mean(x ** 2)) for x in chans]
                max_rms = max(chan_rms)
            else:  # 'windowed'
                win_length = int(self.fs * 0.01)  # 10ms running window
                chan_rms = [running_rms(x, win_length) for x in chans]
                max_rms = max([max(x) for x in chan_rms])
            if max_rms > 2 * self._stim_rms:
                warn_string = ('Expyfun: Stimulus max RMS ({}) exceeds stated '
                               'RMS ({}) by more than 6 dB.'
                               ''.format(max_rms, self._stim_rms))
                logger.warning(warn_string)
                warnings.warn(warn_string)
            elif max_rms < 0.5 * self._stim_rms:
                warn_string = ('Expyfun: Stimulus max RMS ({}) is less than '
                               'stated RMS ({}) by more than 6 dB.'
                               ''.format(max_rms, self._stim_rms))
                logger.warning(warn_string)

        # always prepend a zero to deal with TDT reset of buffer position
        samples = np.r_[np.atleast_2d([0.0, 0.0]), samples]
        return np.ascontiguousarray(samples)

    def set_rms_checking(self, check_rms):
        """Set the RMS checking flag.

        Parameters
        ----------
        check_rms : str | None
            Method to use in checking stimulus RMS to ensure appropriate
            levels. ``'windowed'`` uses a 10ms window to find the max RMS in
            each channel and checks to see that it is within 6 dB of the stated
            ``stim_rms``.  ``'wholefile'`` checks the RMS of the stimulus as a
            whole, while ``None`` disables RMS checking.
        """
        if check_rms not in [None, 'wholefile', 'windowed']:
            raise ValueError('check_rms must be one of "wholefile", "windowed"'
                             ', or None.')
        self._check_rms = check_rms

################################ OTHER METHODS ###############################
    def write_data_line(self, event_type, value=None, timestamp=None):
        """Add a line of data to the output CSV.

        Parameters
        ----------
        event_type : str
            Type of event (e.g., keypress, screen flip, etc.)
        value : None | str
            Anything that can be cast to a string is okay here.
        timestamp : float | None
            The timestamp when the event occurred.  If ``None``, will use the
            time the data line was written from the master clock.

        Notes
        -----
        Writing a data line does not cause the file to be flushed.
        """
        if timestamp is None:
            timestamp = self._master_clock()
        ll = '\t'.join(_sanitize(x) for x in [timestamp, event_type,
                                              value]) + '\n'
        if self._data_file is not None:
            self._data_file.write(ll)

    def _get_time_correction(self, clock_type):
        """Clock correction (seconds) for win.flip().
        """
        time_correction = (self._master_clock() -
                           self._time_correction_fxns[clock_type]())
        if clock_type not in self._time_corrections:
            self._time_corrections[clock_type] = time_correction

        diff = time_correction - self._time_corrections[clock_type]
        if np.abs(diff) > 10e-6:
            logger.warning('Expyfun: drift of > 10 microseconds ({}) '
                           'between {} clock and EC master clock.'
                           ''.format(round(diff * 10e6), clock_type))
        logger.debug('Expyfun: time correction between {} clock and EC '
                     'master clock is {}. This is a change of {}.'
                     ''.format(clock_type, time_correction, time_correction
                               - self._time_corrections[clock_type]))
        return time_correction

    def wait_secs(self, *args, **kwargs):
        """Wait a specified number of seconds.

        Parameters
        ----------
        secs : float
            Number of seconds to wait.
        hog_cpu_time : float
            Amount of CPU time to hog. See Notes.

        Notes
        -----
        See the wait_secs() function.
        """
        wait_secs(*args, **kwargs)

    def wait_until(self, timestamp):
        """Wait until the given time is reached.

        Parameters
        ----------
        timestamp : float
            A time to wait until, evaluated against the experiment master
            clock.

        Returns
        -------
        remaining_time : float
            The difference between ``timestamp`` and the time ``wait_until``
            was called.

        Notes
        -----
        Unlike ``wait_secs``, there is no guarantee of precise timing with this
        function. It is the responsibility of the user to do choose a
        reasonable timestamp (or equivalently, do a reasonably small amount of
        processing prior to calling ``wait_until``).
        """
        time_left = timestamp - self._master_clock()
        if time_left < 0:
            logger.warning('Expyfun: wait_until was called with a timestamp '
                           '({}) that had already passed {} seconds prior.'
                           ''.format(timestamp, -time_left))
        else:
            wait_secs(time_left)
        return time_left

    def identify_trial(self, **ids):
        """Identify trial type before beginning the trial

        Parameters
        ----------
        **ids : keyword arguments
            Ids to stamp, e.g. ``ec_id='TL90,MR45'. Use ``ec.id_types``
            to see valid options. Typical choices are ``ec_id``, ``el_id``,
            and ``ttl_id`` for experiment controller, eyelink, and TDT
            (or parallel port) respectively.
        """
        if self._trial_identified:
            raise RuntimeError('Cannot identify a trial twice')
        call_set = set(self._id_call_dict.keys())
        passed_set = set(ids.keys())
        if not call_set == passed_set:
            raise KeyError('All keys passed in {0} must match the set of '
                           'keys required {1}'.format(passed_set, call_set))
        ll = max([len(key) for key in ids.keys()])
        for key, id_ in ids.items():
            logger.exp('Expyfun: Stamp trial ID to {0} : {1}'
                       ''.format(key.ljust(ll), str(id_)))
            self._id_call_dict[key](id_)
        self._trial_identified = True

    def trial_ok(self):
        """Report that the trial was okay and do post-trial tasks.

        For example, logs and data files can be flushed at the end of each
        trial.
        """
        for func in self._on_trial_ok:
            func()

    def _stamp_ec_id(self, id_):
        """Stamp id -- currently anything allowed"""
        self.write_data_line('trial_id', id_)

    def _stamp_binary_id(self, id_):
        """Helper for ec to stamp a set of IDs using binary controller

        This makes TDT and parallel port give the same output. Eventually
        we may want to customize it so that parallel could work differently,
        but for now it's unified."""
        if not isinstance(id_, (list, tuple, np.ndarray)):
            raise TypeError('id must be array-like')
        id_ = np.array(id_)
        if not np.all(np.logical_or(id_ == 1, id_ == 0)):
            raise ValueError('All values of id must be 0 or 1')
        id_ = 2 ** (id_.astype(int) + 2)  # 4's and 8's
        # Put 8, 8 on ends
        id_ = np.concatenate(([8], id_, [8]))
        self._stamp_ttl_triggers(id_)

    def _stamp_ttl_triggers(self, ids):
        """Helper to stamp triggers without input checking"""
        self._ttl_stamp_func(ids)

    def flush(self):
        """Flush logs and data files
        """
        flush_logger()
        if self._data_file is not None and not self._data_file.closed:
            self._data_file.flush()

    def close(self):
        """Close all connections in experiment controller.
        """
        self.__exit__(None, None, None)

    def __enter__(self):
        logger.debug('Expyfun: Entering')
        return self

    def __exit__(self, err_type, value, traceback):
        """
        Notes
        -----
        err_type, value and traceback will be None when called by self.close()
        """
        logger.debug('Expyfun: Exiting cleanly')

        # do external cleanups
        cleanup_actions = [self.stop_noise, self.stop]
        cleanup_actions.extend(self._extra_cleanup_fun)
        if hasattr(self, '_win'):
            # do this last, as other methods may add/remove handlers
            cleanup_actions.append(self._win.close)
        for action in cleanup_actions:
            try:
                action()
            except Exception:
                tb.print_exc()
                pass

        # clean up our API
        try:
            self.flush()
        except Exception:
            tb.print_exc()
            pass

        if any([x is not None for x in (err_type, value, traceback)]):
            return False
        return True

############################# READ-ONLY PROPERTIES ###########################
    @property
    def id_types(self):
        """Trial ID types needed for each trial"""
        return list(self._id_call_dict.keys())

    @property
    def fs(self):
        """Playback frequency of the audio controller (samples / second).
        """
        return self._ac.fs  # not user-settable

    @property
    def stim_fs(self):
        """Sampling rate at which the stimuli were generated.
        """
        return self._stim_fs  # not user-settable

    @property
    def stim_db(self):
        """Sound power in dB of the stimuli.
        """
        return self._stim_db  # not user-settable

    @property
    def noise_db(self):
        """Sound power in dB of the background noise.
        """
        return self._noise_db  # not user-settable

    @property
    def current_time(self):
        """Timestamp from the experiment master clock.
        """
        return self._master_clock()

    @property
    def _fs_mismatch(self):
        """Quantify if sample rates substantively differ.
        """
        return not np.allclose(self.stim_fs, self.fs, rtol=0, atol=0.5)


def _get_items(d, fixed, title):
    """Helper to get items for an experiment"""
    print(title)
    for key, val in d.iteritems():
        if key in fixed:
            print('{0}: {1}'.format(key, val))
        else:
            d[key] = input('{0}: '.format(key))


def _get_dev_db(audio_controller):
    """Selects device-specific amplitude to ensure equivalence across devices.
    """
    if audio_controller == 'RM1':
        return 108  # this is approx w/ knob @ 12 o'clock (knob not detented)
    elif audio_controller == 'RP2':
        return 108
    elif audio_controller == 'RZ6':
        return 114
    elif audio_controller == 'pyglet':
        return 90  # TODO: this value not yet calibrated, may vary by system
    elif audio_controller == 'dummy':  # only used for testing
        return 90
    else:
        logger.warning('Expyfun: Unknown audio controller: stim scaler may '
                       'not work correctly. You may want to remove your '
                       'headphones if this is the first run of your '
                       'experiment.')
        return 90  # for untested TDT models
