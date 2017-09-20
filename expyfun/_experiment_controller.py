"""Tools for controlling experiment execution"""

# Authors: Dan McCloy <drmccloy@uw.edu>
#          Eric Larson <larsoner@uw.edu>
#          Jasper van den Bosch <jasperb@uw.edu>
#
# License: BSD (3-clause)

import numpy as np
import os
import warnings
from os import path as op
from functools import partial
import traceback as tb

from ._utils import (get_config, verbose_dec, _check_pyglet_version, wait_secs,
                     running_rms, _sanitize, logger, ZeroClock, date_str,
                     check_units, set_log_file, flush_logger,
                     string_types, _fix_audio_dims, input)
from ._tdt_controller import TDTController
from ._trigger_controllers import ParallelTrigger
from ._sound_controllers import PygletSoundController, SoundPlayer
from ._input_controllers import Keyboard, CedrusBox, Mouse
from .visual import Text, Rectangle, Video, _convert_color
from ._git import assert_version

# Note: ec._trial_progress has three values:
# 1. 'stopped', which ec.identify_trial turns into...
# 2. 'identified', which ec.start_stimulus turns into...
# 3. 'started', which ec.trial_ok turns into 'stopped'.


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
        (see documentation for :class:`TDTController`).
    response_device : str | None
        Must be 'keyboard', 'cedrus', or 'tdt'.  If None, the type will be read
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
        created. Data will be saved to ``output_dir/SUBJECT_DATE``.
        If None, no output data or logs will be saved (ONLY FOR TESTING!).
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
        ``['lctrl', 'rctrl']``.  Using ``['escape']`` is not recommended due to
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
    check_rms : str | None
        Method to use in checking stimulus RMS to ensure appropriate levels.
        Possible values are ``None``, ``wholefile``, and ``windowed`` (the
        default); see `set_rms_checking` for details.
    suppress_resamp : bool
        If ``True``, will suppress resampling of stimuli to the sampling
        frequency of the sound output device.
    version : str | None
        A length-7 string passed to ``expyfun.assert_version()`` to ensure that
        the expected version of the expyfun codebase is being used when running
        experiments. To override version checking (e.g., during development)
        use ``version='dev'``.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see expyfun.verbose).

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
                 stim_rms=0.01, stim_fs=24414, stim_db=65, noise_db=45,
                 output_dir='data', window_size=None, screen_num=None,
                 full_screen=True, force_quit=None, participant=None,
                 monitor=None, trigger_controller=None, session=None,
                 check_rms='windowed', suppress_resamp=False, version=None,
                 enable_video=False, verbose=None):
        # initialize some values
        self._stim_fs = stim_fs
        self._stim_rms = stim_rms
        self._stim_db = stim_db
        self._noise_db = noise_db
        self._stim_scaler = None
        self._suppress_resamp = suppress_resamp
        self._enable_video = enable_video
        self.video = None
        self._bgcolor = _convert_color('k')
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
                logger.warning('Expyfun: using "escape" as a force-quit key '
                               'is not recommended because it has special '
                               'status in pyglet.')
            # check expyfun version
            if version is None:
                raise RuntimeError('You must specify an expyfun version string'
                                   ' to use ExperimentController, or specify '
                                   'version=\'dev\' to override.')
            elif version.lower() != 'dev':
                assert_version(version)
            # set up timing
            # Use ZeroClock, which uses the "clock" fn but starts at zero
            self._time_corrections = dict()
            self._time_correction_fxns = dict()
            self._time_correction_maxs = dict()  # optional, defaults to 10e-6

            # dictionary for experiment metadata
            self._exp_info = {'participant': participant, 'session': session,
                              'exp_name': exp_name, 'date': date_str()}

            # session start dialog, if necessary
            fixed_list = ['exp_name', 'date']  # things not editable in GUI
            for key, value in self._exp_info.items():
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
            self._output_dir = None
            set_log_file(None)
            if output_dir is not None:
                output_dir = op.abspath(output_dir)
                if not op.isdir(output_dir):
                    os.mkdir(output_dir)
                basename = op.join(output_dir, '{}_{}'
                                   ''.format(self._exp_info['participant'],
                                             self._exp_info['date']))
                self._output_dir = basename
                self._log_file = self._output_dir + '.log'
                set_log_file(self._log_file)
                closer = partial(set_log_file, None)
                self._extra_cleanup_fun.append(closer)
                # initialize data file
                self._data_file = open(self._output_dir + '.tab', 'a')
                self._extra_cleanup_fun.append(self._data_file.close)
                self._data_file.write('# ' + str(self._exp_info) + '\n')
                self.write_data_line('event', 'value', 'timestamp')

            #
            # set up monitor
            #
            if screen_num is None:
                screen_num = int(get_config('SCREEN_NUM', '0'))
            if monitor is None:
                import pyglet
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
            monitor['SCREEN_HEIGHT'] = (monitor['SCREEN_WIDTH'] /
                                        float(monitor['SCREEN_SIZE_PIX'][0]) *
                                        float(monitor['SCREEN_SIZE_PIX'][1]))
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
                                     ' \'pyglet\' or \'TYPE\': \'tdt\').')
            elif not isinstance(audio_controller, dict):
                raise TypeError('audio_controller must be a str or dict.')
            self.audio_type = audio_controller['TYPE'].lower()

            #
            # parse response device
            #
            if response_device is None:
                response_device = get_config('RESPONSE_DEVICE', 'keyboard')
            if response_device not in ['keyboard', 'tdt', 'cedrus']:
                raise ValueError('response_device must be "keyboard", "tdt", '
                                 '"cedrus", or None')
            self._response_device = response_device

            #
            # Initialize devices
            #

            # Audio (and for TDT, potentially keyboard)
            if self.audio_type == 'tdt':
                logger.info('Expyfun: Setting up TDT')
                self._ac = TDTController(audio_controller)
                self.audio_type = self._ac.model
            elif self.audio_type == 'pyglet':
                self._ac = PygletSoundController(self, self.stim_fs)
            else:
                raise ValueError('audio_controller[\'TYPE\'] must be '
                                 '\'pyglet\' or \'tdt\'.')
            self._extra_cleanup_fun.append(self._ac.halt)
            # audio scaling factor; ensure uniform intensity across devices
            self.set_stim_db(self._stim_db)
            self.set_noise_db(self._noise_db)

            if self._fs_mismatch:
                msg = ('Expyfun: Mismatch between reported stim sample '
                       'rate ({0}) and device sample rate ({1}).'
                       ''.format(self.stim_fs, self.fs))
                if self._suppress_resamp:
                    msg += ('Nothing will be done about this because '
                            'suppress_resamp is "True"')
                else:
                    msg += ('Experiment Controller will resample for you, but '
                            'this takes a non-trivial amount of processing '
                            'time and may compromise your experimental '
                            'timing and/or cause artifacts.')
                logger.warning(msg)

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
            elif response_device == 'tdt':
                if not isinstance(self._ac, TDTController):
                    raise ValueError('response_device can only be "tdt" if '
                                     'tdt is used for audio')
                self._response_handler = self._ac
                self._ac._add_keyboard_init(self, force_quit)
            else:  # response_device == 'cedrus'
                self._response_handler = CedrusBox(self, force_quit)

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
                self._stamp_ttl_triggers = self._ac.stamp_triggers
            elif trigger_controller['type'] in ['parallel', 'dummy']:
                if 'address' not in trigger_controller['type']:
                    addr = get_config('TRIGGER_ADDRESS')
                    trigger_controller['address'] = addr
                out = ParallelTrigger(trigger_controller['type'],
                                      trigger_controller.get('address'))
                self._stamp_ttl_triggers = out.stamp_triggers
                self._extra_cleanup_fun.append(out.close)
            else:
                raise ValueError('trigger_controller type must be '
                                 '"parallel", "dummy", or "tdt", not '
                                 '{0}'.format(trigger_controller['type']))
            self._id_call_dict['ttl_id'] = self._stamp_binary_id

            # other basic components
            self._mouse_handler = Mouse(self)
            t = np.arange(44100 // 3) / 44100.
            car = sum([np.sin(2 * np.pi * f * t) for f in [800, 1000, 1200]])
            self._beep = None
            self._beep_data = np.tile(car * np.exp(-t * 10) / 4, (2, 3))

            # finish initialization
            logger.info('Expyfun: Initialization complete')
            logger.exp('Expyfun: Subject: {0}'
                       ''.format(self._exp_info['participant']))
            logger.exp('Expyfun: Session: {0}'
                       ''.format(self._exp_info['session']))
            ok_log = partial(self.write_data_line, 'trial_ok', None)
            self._on_trial_ok.append(ok_log)
            self._on_trial_ok.append(self.flush)
            self._trial_progress = 'stopped'
            self._ofp_critical_funs = list()
        except Exception:
            self.close()
            raise
        # hack to prevent extra flips on first screen_prompt / screen_text
        self.flip()

    def __repr__(self):
        """Return a useful string representation of the experiment
        """
        string = ('<ExperimentController ({3}): "{0}" {1} ({2})>'
                  ''.format(self._exp_info['exp_name'],
                            self._exp_info['participant'],
                            self._exp_info['session'],
                            self.audio_type))
        return string

# ############################### SCREEN METHODS ##############################
    def screen_text(self, text, pos=[0, 0], color='white', font_name='Arial',
                    font_size=24, wrap=True, units='norm', attr=True):
        """Show some text on the screen.

        Parameters
        ----------
        text : str
            The text to be rendered.
        pos : list | tuple
            x, y position of the text. In the default units (-1 to 1, with
            positive going up and right) the default is dead center (0, 0).
        color : matplotlib color
            The text color.
        font_name : str
            The name of the font to use.
        font_size : float
            The font size (in points) to use.
        wrap : bool
            Whether or not the text will wrap to fit in screen, appropriate
            for multi-line text. Inappropriate for text requiring
            precise positioning or centering.
        units : str
            Units for `pos`. See `check_units` for options. Applies to
            `pos` but not `font_size`.
        attr : bool
            Should the text be interpreted with pyglet's ``decode_attributed``
            method? This allows inline formatting for text color, e.g.,
            ``'This is {color (255, 0, 0, 255)}red text'``.

        Returns
        -------
        Instance of visual.Text

        See Also
        --------
        ExperimentController.screen_prompt
        """
        check_units(units)
        scr_txt = Text(self, text, pos, color, font_name, font_size,
                       wrap=wrap, units=units, attr=attr)
        scr_txt.draw()
        self.call_on_next_flip(partial(self.write_data_line, 'screen_text',
                                       text))
        return scr_txt

    def screen_prompt(self, text, max_wait=np.inf, min_wait=0, live_keys=None,
                      timestamp=False, clear_after=True, pos=[0, 0],
                      color='white', font_name='Arial', font_size=24,
                      wrap=True, units='norm', attr=True):
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
        pos : list | tuple
            x, y position of the text. In the default units (-1 to 1, with
            positive going up and right) the default is dead center (0, 0).
        color : matplotlib color
            The text color.
        font_name : str
            The name of the font to use.
        font_size : float
            The font size (in points) to use.
        wrap : bool
            Whether or not the text will wrap to fit in screen, appropriate
            for multi-line text. Inappropriate for text requiring
            precise positioning or centering.
        units : str
            Units for `pos`. See `check_units` for options. Applies to
            `pos` but not `font_size`.
        attr : bool
            Should the text be interpreted with pyglet's ``decode_attributed``
            method? This allows inline formatting for text color, e.g.,
            ``'This is {color (255, 0, 0, 255)}red text'``.

        Returns
        -------
        pressed : tuple | str | None
            If ``timestamp==True``, returns a tuple ``(str, float)`` indicating
            the first key pressed and its timestamp (or ``(None, None)`` if no
            acceptable key was pressed between ``min_wait`` and ``max_wait``).
            If ``timestamp==False``, returns a string indicating the first key
            pressed (or ``None`` if no acceptable key was pressed).

        See Also
        --------
        ExperimentController.screen_text
        """
        if not isinstance(text, list):
            text = [text]
        if not all([isinstance(t, string_types) for t in text]):
            raise TypeError('text must be a string or list of strings')
        for t in text:
            self.screen_text(t, pos=pos, color=color, font_name=font_name,
                             font_size=font_size, wrap=wrap, units=units,
                             attr=attr)
            self.flip()
            out = self.wait_one_press(max_wait, min_wait, live_keys,
                                      timestamp)
        if clear_after:
            self.flip()
        return out

    def set_background_color(self, color='black'):
        """Set and draw a solid background color

        Parameters
        ----------
        color : matplotlib color
            The background color.

        Notes
        -----
        This should be called before anything else is drawn to the buffer,
        since it will draw a filled rectangle over everything. On subsequent
        flips, the rectangle will automatically be "drawn" because
        ``glClearColor`` will be set so the buffer starts out with the
        appropriate backgound color.
        """
        from pyglet import gl
        if not self._enable_video:
            # we go a little over here to be safe from round-off errors
            Rectangle(self, pos=[0, 0, 2.1, 2.1], fill_color=color).draw()
        self._bgcolor = _convert_color(color)
        gl.glClearColor(*[c / 255. for c in self._bgcolor])

    def start_stimulus(self, start_of_trial=True, flip=True, when=None):
        """Play audio, (optionally) flip screen, run any "on_flip" functions.

        Parameters
        ----------
        start_of_trial : bool
            If True, it checks to make sure that the trial ID has been
            stamped appropriately. Set to False only in cases where
            ``flip_and_play`` is to be used mid-trial (should be rare!).
        flip : bool
            If False, don't flip the screen.
        when : float | None
            Time to start stimulus. If None, start immediately.
            Note that due to flip timing limitations, this is only a
            guaranteed *minimum* (not absolute) wait time before the
            flip completes (if `flip` is ``True``). As a result, in some
            cases `when` should be set to a value smaller than your true
            intended flip time.

        Returns
        -------
        flip_time : float
            The timestamp of the screen flip.

        See Also
        --------
        ExperimentController.identify_trial
        ExperimentController.flip
        ExperimentController.play
        ExperimentController.stop
        ExperimentController.trial_ok

        Notes
        -----
        Order of operations is: screen flip (optional), audio start, then
        (only if ``flip=True``) additional functions added with
        `call_on_next_flip` and `call_on_every_flip`.
        """
        if start_of_trial:
            if self._trial_progress != 'identified':
                raise RuntimeError('Trial ID must be stamped before starting '
                                   'the trial')
            self._trial_progress = 'started'
        extra = 'flipping screen and ' if flip else ''
        logger.exp('Expyfun: Starting stimuli: {0}playing audio'.format(extra))
        # ensure self._play comes first in list, followed by other critical
        # private functions (e.g., EL stamping), then user functions:
        if flip:
            self._on_next_flip = ([self._play] + self._ofp_critical_funs +
                                  self._on_next_flip)
            stimulus_time = self.flip(when)
        else:
            if when is not None:
                self.wait_until(when)
            funs = [self._play] + self._ofp_critical_funs
            self._win.dispatch_events()
            stimulus_time = self.get_time()
            for fun in funs:
                fun()
        return stimulus_time

    def call_on_next_flip(self, function):
        """Add a function to be executed on next flip only.

        Parameters
        ----------
        function : function | None
            The function to call. If ``None``, all the "on every flip"
            functions will be cleared.

        See Also
        --------
        ExperimentController.call_on_every_flip

        Notes
        -----
        See `flip_and_play` for order of operations. Can be called multiple
        times to add multiple functions to the queue. If the function must be
        called with arguments, use `functools.partial` before passing to
        `call_on_next_flip`.
        """
        if function is not None:
            self._on_next_flip.append(function)
        else:
            self._on_next_flip = []

    def call_on_every_flip(self, function):
        """Add a function to be executed on every flip.

        Parameters
        ----------
        function : function | None
            The function to call. If ``None``, all the "on every flip"
            functions will be cleared.

        See Also
        --------
        ExperimentController.call_on_next_flip

        Notes
        -----
        See `flip_and_play` for order of operations. Can be called multiple
        times to add multiple functions to the queue. If the function must be
        called with arguments, use `functools.partial` before passing to
        `call_on_every_flip`.
        """
        if function is not None:
            self._on_every_flip.append(function)
        else:
            self._on_every_flip = []

    def _convert_units(self, verts, fro, to):
        """Convert between different screen units"""
        check_units(to)
        check_units(fro)
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
        import pyglet
        # this must be done in order to instantiate image_buffer_manager
        data = pyglet.image.get_buffer_manager().get_color_buffer()
        data = self._win.context.image_buffer_manager.color_buffer.image_data
        data = data.get_data(data.format, data.pitch)
        data = np.fromstring(data, dtype=np.uint8)
        data.shape = (self._win.height, self._win.width, 4)
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

# ############################### VIDEO METHODS ###############################
    def load_video(self, file_name, pos=(0, 0), units='norm', center=True):
        from pyglet.media import MediaFormatException
        try:
            self.video = Video(self, file_name, pos, units)
        except MediaFormatException:
            err = ('Something is wrong; probably you tried to load a '
                   'compressed video file but you do not have AVbin installed.'
                   ' Download and install it; if you are on Windows, you may '
                   'also need to manually copy the AVbin .dll file(s) from '
                   'C:\Windows\system32 to C:\Windows\SysWOW64.')
            raise RuntimeError(err)

    def delete_video(self):
        self.video._delete()
        self.video = None

# ############################### OPENGL METHODS ##############################
    def _setup_window(self, window_size, exp_name, full_screen, screen_num):
        import pyglet
        from pyglet import gl
        # Use 16x sampling here
        config_kwargs = dict(depth_size=8, double_buffer=True, stereo=False,
                             stencil_size=0, samples=0, sample_buffers=0)
        # Travis can't handle multi-sampling, but our production machines must
        if os.getenv('TRAVIS') == 'true':
            del config_kwargs['samples'], config_kwargs['sample_buffers']
        self._full_screen = full_screen
        win_kwargs = dict(width=window_size[0], height=window_size[1],
                          caption=exp_name, fullscreen=False,
                          screen=screen_num, style='borderless', visible=False,
                          config=pyglet.gl.Config(**config_kwargs))

        max_try = 5  # sometimes it fails for unknown reasons
        for ii in range(max_try):
            try:
                win = pyglet.window.Window(**win_kwargs)
            except pyglet.gl.ContextException:
                if ii == max_try - 1:
                    raise
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
        gl.glShadeModel(gl.GL_SMOOTH)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        v_ = False if os.getenv('_EXPYFUN_WIN_INVISIBLE') == 'true' else True
        self.set_visible(v_)
        win.dispatch_events()

    def flip(self, when=None):
        """Flip screen, then run any "on-flip" functions.

        Parameters
        ----------
        when : float | None
            Time to flip. If None, flip immediately. Note that due to flip
            timing limitations, this is only a guaranteed *minimum* (not
            absolute) wait time before the flip completes. As a result, in
            some cases `when` should be set to a value smaller than your
            true intended flip time.

        Returns
        -------
        flip_time : float
            The timestamp of the screen flip.

        See Also
        --------
        ExperimentController.identify_trial
        ExperimentController.play
        ExperimentController.start_stimulus
        ExperimentController.stop
        ExperimentController.trial_ok

        Notes
        -----
        Order of operations is: screen flip, functions added with
        `call_on_next_flip`, followed by functions added with
        `call_on_every_flip`.
        """
        from pyglet import gl
        if when is not None:
            self.wait_until(when)
        call_list = self._on_next_flip + self._on_every_flip
        self._win.dispatch_events()
        self._win.switch_to()
        gl.glFinish()
        self._win.flip()
        # this waits until everything is called, including last draw
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glBegin(gl.GL_POINTS)
        if not self._enable_video:
            gl.glColor4f(0, 0, 0, 0)
        gl.glVertex2i(10, 10)
        gl.glEnd()
        gl.glFinish()
        flip_time = self.get_time()
        for function in call_list:
            function()
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

    def set_visible(self, visible=True, flip=True):
        """Set the window visibility

        Parameters
        ----------
        visible : bool
            The visibility.
        flip : bool
            If `visible` is ``True``, call `flip` after setting visible.
            This fixes an issue with the window background color not
            being set properly for the first draw after setting visible;
            by default (at least on Linux) the background is black when
            the window is restored, regardless of what the glClearColor
            had been set to.
        """
        self._win.set_fullscreen(visible and self._full_screen)
        self._win.set_visible(visible)
        logger.exp('Expyfun: Set screen visibility {0}'.format(visible))
        if visible and flip:
            self.flip()

# ############################## KEYPRESS METHODS #############################
    def listen_presses(self):
        """Start listening for keypresses.

        See Also
        --------
        ExperimentController.get_presses
        ExperimentController.wait_one_press
        ExperimentController.wait_for_presses
        """
        self._response_handler.listen_presses()

    def get_presses(self, live_keys=None, timestamp=True, relative_to=None,
                    kind='presses', return_kinds=False):
        """Get the entire keyboard / button box buffer.

        This will also clear events that are not requested per ``type``.

        .. warning:

            It is currently not possible to get key-release events for Cedrus
            boxes or TDT. Therefore, using get_presses(type='releases') or
            get_presses(type='both') will throw an exception.

        Parameters
        ----------
        live_keys : list | None
            List of strings indicating acceptable keys or buttons. Other data
            types are cast as strings, so a list of ints will also work.
            ``None`` accepts all keypresses.
        timestamp : bool
            Whether the keypress should be timestamped. If True, returns the
            button press time relative to the value given in `relative_to`.
        relative_to : None | float
            A time relative to which timestamping is done. Ignored if
            timestamp==False.  If ``None``, timestamps are relative to the time
            `listen_presses` was last called.
        kind : string
            Which key events to return. One of ``presses``, ``releases`` or
            ``both``. (default ``presses``)
        return_kinds : bool
            Return the kinds of presses.

        Returns
        -------
        presses : list
            Returns a list of tuples with key events. Each tuple's first value
            will be the key pressed. If ``timestamp==True``, the second value
            is the time for the event. If ``return_kinds==True``, then the
            last value is a string indicating if this was a key press or
            release event.

        See Also
        --------
        ExperimentController.listen_presses
        ExperimentController.wait_one_press
        ExperimentController.wait_for_presses
        """
        return self._response_handler.get_presses(
            live_keys, timestamp, relative_to, kind, return_kinds)

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
            ``None`` accepts all keypresses.
        timestamp : bool
            Whether the keypress should be timestamped. If ``True``, returns
            the button press time relative to the value given in
            `relative_to`.
        relative_to : None | float
            A time relative to which timestamping is done. Ignored if
            ``timestamp==False``.  If ``None``, timestamps are relative to the
            time `wait_one_press` was called.

        Returns
        -------
        pressed : tuple | str | None
            If ``timestamp==True``, returns a tuple (str, float) indicating the
            first key pressed and its timestamp (or ``(None, None)`` if no
            acceptable key was pressed between `min_wait` and `max_wait`).
            If ``timestamp==False``, returns a string indicating the first key
            pressed (or ``None`` if no acceptable key was pressed).

        See Also
        --------
        ExperimentController.listen_presses
        ExperimentController.get_presses
        ExperimentController.wait_for_presses
        """
        return self._response_handler.wait_one_press(
            max_wait, min_wait, live_keys, timestamp, relative_to)

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
            ``None`` accepts all keypresses.
        timestamp : bool
            Whether the keypresses should be timestamped. If ``True``, returns
            the button press time relative to the value given in
            `relative_to`.
        relative_to : None | float
            A time relative to which timestamping is done. Ignored if
            `timestamp` is ``False``.  If ``None``, timestamps are relative
            to the time `wait_for_presses` was called.

        Returns
        -------
        presses : list
            If timestamp==False, returns a list of strings indicating which
            keys were pressed. Otherwise, returns a list of tuples
            (str, float) of keys and their timestamps.

        See Also
        --------
        ExperimentController.listen_presses
        ExperimentController.get_presses
        ExperimentController.wait_one_press
        """
        return self._response_handler.wait_for_presses(
            max_wait, min_wait, live_keys, timestamp, relative_to)

    def _log_presses(self, pressed):
        """Write key presses to data file."""
        # This function will typically be called by self._response_handler
        # after it retrieves some button presses
        for key, stamp, eventType in pressed:
            self.write_data_line('key'+eventType, key, stamp)

    def check_force_quit(self):
        """Check to see if any force quit keys were pressed."""
        self._response_handler.check_force_quit()
    
    def check_interrupts(self):
        """Check to see if any interrupt keys were pressed."""
        self._response_handler.check_interrupts()

# ############################## MOUSE METHODS ################################
    def listen_clicks(self):
        """Start listening for mouse clicks.

        See Also
        --------
        ExperimentController.get_clicks
        ExperimentController.get_mouse_position
        ExperimentController.toggle_cursor
        """
        self._mouse_handler.listen_clicks()

    def get_clicks(self, live_buttons=None, timestamp=True, relative_to=None):
        """Get the entire keyboard / button box buffer.

        Parameters
        ----------
        live_buttons : list | None
            List of strings indicating acceptable buttons.
            ``None`` accepts all mouse clicks.
        timestamp : bool
            Whether the mouse click should be timestamped. If True, returns the
            button click time relative to the value given in `relative_to`.
        relative_to : None | float
            A time relative to which timestamping is done. Ignored if
            timestamp==False.  If ``None``, timestamps are relative to the time
            `listen_clicks` was last called.

        Returns
        -------
        clicks : list of tuple
            Returns a list of the clicks between min_wait and max_wait.
            If ``timestamp==True``, each entry is a tuple (str, int, int,
            float) indicating the button clicked and its timestamp.
            If ``timestamp==False``, each entry is a tuple (str, int, int)
            indicating the button clicked.

        See Also
        --------
        ExperimentController.get_mouse_position
        ExperimentController.listen_clicks
        ExperimentController.toggle_cursor
        ExperimentController.wait_one_click
        ExperimentController.wait_for_clicks
        """
        return self._mouse_handler.get_clicks(live_buttons, timestamp,
                                              relative_to)

    def get_mouse_position(self, units='pix'):
        """Mouse position in screen coordinates

        Parameters
        ----------
        units : str
            Units to return. See `check_units` for options.

        Returns
        -------
        position : ndarray
            The mouse position.

        See Also
        --------
        ExperimentController.get_clicks
        ExperimentController.listen_clicks
        ExperimentController.toggle_cursor
        ExperimentController.wait_one_click
        ExperimentController.wait_for_clicks
        """
        check_units(units)
        pos = np.array(self._mouse_handler.pos)
        pos = self._convert_units(pos[:, np.newaxis], 'norm', units)[:, 0]
        return pos

    def toggle_cursor(self, visibility, flip=False):
        """Show or hide the mouse

        Parameters
        ----------
        visibility : bool
            If True, show; if False, hide.

        See Also
        --------
        ExperimentController.get_clicks
        ExperimentController.get_mouse_position
        ExperimentController.listen_clicks
        ExperimentController.wait_one_click
        ExperimentController.wait_for_clicks
        """
        try:
            self._mouse_handler.set_visible(visibility)
            # TODO move mouse to lower right corner for windows no-hide bug
        except Exception:
            pass  # pyglet bug on Linux!
        if flip:
            self.flip()

    def wait_one_click(self, max_wait=np.inf, min_wait=0.0, live_buttons=None,
                       timestamp=True, relative_to=None, visible=None):
        """Returns only the first mouse button clicked after min_wait.

        Parameters
        ----------
        max_wait : float
            Duration after which control is returned if no button is clicked.
        min_wait : float
            Duration for which to ignore button clicks.
        live_buttons : list | None
            List of strings indicating acceptable buttons.
            ``None`` accepts all mouse clicks.
        timestamp : bool
            Whether the mouse click should be timestamped. If ``True``, returns
            the mouse click time relative to the value given in
            ``relative_to``.
        relative_to : None | float
            A time relative to which timestamping is done. Ignored if
            ``timestamp==False``.  If ``None``, timestamps are relative to the
            time `wait_one_click` was called.
        visible : None | bool
            Whether to show the cursor while in the function. ``None`` has no
            effect and is the default. A boolean will show it (or not) while
            the function has control and then set visibility back to its
            previous value afterwards.

        Returns
        -------
        clicked : tuple | str | None
            If ``timestamp==True``, returns a tuple (str, int, int, float)
            indicating the first button clicked and its timestamp (or
            ``(None, None, None, None)`` if no acceptable button was clicked
            between `min_wait` and `max_wait`). If ``timestamp==False``,
            returns a tuple (str, int, int) indicating the first button clicked
            (or ``(None, None, None)`` if no acceptable key was clicked).

        See Also
        --------
        ExperimentController.get_clicks
        ExperimentController.get_mouse_position
        ExperimentController.listen_clicks
        ExperimentController.toggle_cursor
        ExperimentController.wait_for_clicks
        """
        return self._mouse_handler.wait_one_click(max_wait, min_wait,
                                                  live_buttons, timestamp,
                                                  relative_to, visible)

    def wait_for_clicks(self, max_wait=np.inf, min_wait=0.0, live_buttons=None,
                        timestamp=True, relative_to=None, visible=None):
        """Returns all clicks between min_wait and max_wait.

        Parameters
        ----------
        max_wait : float
            Duration after which control is returned if no button is clicked.
        min_wait : float
            Duration for which to ignore button clicks.
        live_buttons : list | None
            List of strings indicating acceptable buttons.
            ``None`` accepts all mouse clicks.
        timestamp : bool
            Whether the mouse click should be timestamped. If ``True``, returns
            the mouse click time relative to the value given in
            ``relative_to``.
        relative_to : None | float
            A time relative to which timestamping is done. Ignored if
            ``timestamp==False``.  If ``None``, timestamps are relative to the
            time ``wait_one_click`` was called.
        visible : None | bool
            Whether to show the cursor while in the function. ``None`` has no
            effect and is the default. A boolean will show it (or not) while
            the function has control and then set visibility back to its
            previous value afterwards.

        Returns
        -------
        clicks : list of tuple
            Returns a list of the clicks between min_wait and max_wait.
            If ``timestamp==True``, each entry is a tuple (str, int, int,
            float) indicating the button clicked and its timestamp.
            If ``timestamp==False``, each entry is a tuple (str, int, int)
            indicating the button clicked.

        See Also
        --------
        ExperimentController.get_clicks
        ExperimentController.get_mouse_position
        ExperimentController.listen_clicks
        ExperimentController.toggle_cursor
        ExperimentController.wait_one_click
        """
        return self._mouse_handler.wait_for_clicks(max_wait, min_wait,
                                                   live_buttons, timestamp,
                                                   relative_to, visible)

    def wait_for_click_on(self, objects, max_wait=np.inf, min_wait=0.0,
                          live_buttons=None, timestamp=True, relative_to=None):
        """Returns the first click after min_wait over a visual object.

        Parameters
        ----------
        objects : list | Rectangle | Circle
            A list of objects (or a single object) that the user may click on.
            Supported types are: Rectangle, Circle
        max_wait : float
            Duration after which control is returned if no button is clicked.
        min_wait : float
            Duration for which to ignore button clicks.
        live_buttons : list | None
            List of strings indicating acceptable buttons.
            ``None`` accepts all mouse clicks.
        timestamp : bool
            Whether the mouse click should be timestamped. If ``True``, returns
            the mouse click time relative to the value given in
            `relative_to`.
        relative_to : None | float
            A time relative to which timestamping is done. Ignored if
            ``timestamp==False``.  If ``None``, timestamps are relative to the
            time `wait_one_click` was called.

        Returns
        -------
        clicked : tuple | str | None
            If ``timestamp==True``, returns a tuple (str, int, int, float)
            indicating the first valid button clicked and its timestamp (or
            ``(None, None, None, None)`` if no acceptable button was clicked
            between `min_wait` and `max_wait`). If ``timestamp==False``,
            returns a tuple (str, int, int) indicating the first button clicked
            (or ``(None, None, None)`` if no acceptable key was clicked).
        index : the index of the object in the list of objects that was clicked
            on. Returns None if time ran out before a valid click. If objects
            were overlapping, it returns the index of the object that comes
            first in the `objects` argument.
        """
        legal_types = self._mouse_handler._legal_types
        if isinstance(objects, legal_types):
            objects = [objects]
        elif not isinstance(objects, list):
            raise TypeError('objects must be a list or one of: %s' %
                            (legal_types,))
        return self._mouse_handler.wait_for_click_on(
            objects, max_wait, min_wait, live_buttons, timestamp, relative_to)

    def _log_clicks(self, clicked):
        """Write mouse clicks to data file.
        """
        # This function will typically be called by self._response_handler
        # after it retrieves some mouse clicks
        for button, x, y, stamp in clicked:
            self.write_data_line('mouseclick', '%s,%i,%i' % (button, x, y),
                                 stamp)

# ############################## AUDIO METHODS ################################
    def system_beep(self):
        """Play a system beep

        This will play through the system audio, *not* through the
        audio controller (unless that is set to be the system audio).
        This is useful for e.g., notifying that it's time for an
        eye-tracker calibration.
        """
        if self._beep is not None:
            self._beep.delete()
        self._beep = SoundPlayer(self._beep_data, 44100)
        self._beep.play()

    def start_noise(self):
        """Start the background masker noise

        See Also
        --------
        ExperimentController.set_noise_db
        ExperimentController.stop_noise
        """
        self._ac.start_noise()

    def stop_noise(self):
        """Stop the background masker noise

        See Also
        --------
        ExperimentControlller.set_noise_db
        ExperimentController.start_noise
        """
        if self._ac is not None:  # check b/c used by __exit__
            self._ac.stop_noise()

    def load_buffer(self, samples):
        """Load audio data into the audio buffer

        Parameters
        ----------
        samples : np.array
            Audio data as floats scaled to (-1,+1), formatted as numpy array
            with shape (1, N), (2, N), or (N,) dtype float32.

        See Also
        --------
        ExperimentController.play
        ExperimentController.set_stim_db
        ExperimentController.start_stimulus
        ExperimentController.stop
        """
        if self._playing:
            raise RuntimeError('Previous audio must be stopped before loading '
                               'the buffer')
        samples = self._validate_audio(samples)
        samples *= self._stim_scaler
        logger.exp('Expyfun: Loading {} samples to buffer'
                   ''.format(samples.size))
        self._ac.load_buffer(samples)

    def play(self):
        """Start audio playback

        Returns
        -------
        play_time : float
            The timestamp of the audio playback.

        See Also
        --------
        ExperimentController.load_buffer
        ExperimentController.set_stim_db
        ExperimentController.start_stimulus
        ExperimentController.stop
        """
        logger.exp('Expyfun: Playing audio')
        # ensure self._play comes first in list:
        self._play()
        return self.get_time()

    def _play(self):
        """Play the audio buffer.
        """
        if self._playing:
            raise RuntimeError('Previous audio must be stopped before playing')
        self._ac.play()
        logger.debug('Expyfun: started audio')
        self.write_data_line('play')

    @property
    def _playing(self):
        """Whether or not a stimulus is currently playing"""
        return self._ac.playing

    def stop(self):
        """Stop audio buffer playback and reset cursor to beginning of buffer

        See Also
        --------
        ExperimentController.load_buffer
        ExperimentController.play
        ExperimentController.set_stim_db
        ExperimentController.start_stimulus
        """
        if self._ac is not None:  # need to check b/c used in __exit__
            self._ac.stop()
        self.write_data_line('stop')
        logger.exp('Expyfun: Audio stopped and reset.')

    def set_noise_db(self, new_db):
        """Set the level of the background noise

        See Also
        --------
        ExperimentController.start_noise
        ExperimentController.stop_noise
        """
        # Noise is always generated at an RMS of 1
        self._ac.set_noise_level(self._update_sound_scaler(new_db, 1.0))
        self._noise_db = new_db

    def set_stim_db(self, new_db):
        """Set the level of the stimuli

        See Also
        --------
        ExperimentController.load_buffer
        ExperimentController.play
        ExperimentController.start_stimulus
        ExperimentController.stop
        """
        self._stim_db = new_db
        self._stim_scaler = self._update_sound_scaler(new_db, self._stim_rms)
        # not immediate: new value is applied on the next load_buffer call

    def _update_sound_scaler(self, desired_db, orig_rms):
        """Calcs coefficient ensuring stim ampl equivalence across devices.
        """
        exponent = (-(_get_dev_db(self.audio_type) - desired_db) / 20.0)
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
            The correctly formatted audio samples. Will be a copy of
            the original samples.
        """
        # check data type
        samples = np.asarray(samples, dtype=np.float32)

        # check values
        if np.max(np.abs(samples)) > 1:
            raise ValueError('Sound data exceeds +/- 1.')
            # samples /= np.max(np.abs(samples),axis=0)

        # check shape and dimensions, make stereo
        samples = _fix_audio_dims(samples, 2).T

        # This limit is currently set by the TDT SerialBuf objects
        # (per channel), it sets the limit on our stimulus durations...
        if np.isclose(self.stim_fs, 24414, atol=1):
            max_samples = 4000000 - 1
            if samples.shape[0] > max_samples:
                raise RuntimeError('Sample too long {0} > {1}'
                                   ''.format(samples.shape[0], max_samples))

        # resample if needed
        if self._fs_mismatch and not self._suppress_resamp:
            logger.warning('Expyfun: Resampling {} seconds of audio'
                           ''.format(round(len(samples) / self.stim_fs), 2))
            from mne.filter import resample
            samples = resample(samples, self.fs, self.stim_fs, axis=0)

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

        # this will create a copy, so we can modify inplace later!
        samples = np.array(samples, np.float32)
        return samples

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

# ############################## OTHER METHODS ################################
    @property
    def participant(self):
        return self._exp_info['participant']

    @property
    def session(self):
        return self._exp_info['session']

    @property
    def exp_name(self):
        return self._exp_info['exp_name']

    @property
    def data_fname(self):
        """Date filename"""
        return self._data_file.name

    def get_time(self):
        """Return current master clock time

        Returns
        -------
        time : float
            Time since ExperimentController was created.
        """
        return self._clock.get_time()

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
        if self._data_file is not None and not self._data_file.closed:
            self._data_file.write(ll)

    def _get_time_correction(self, clock_type):
        """Clock correction (sec) for different devices (screen, bbox, etc.)
        """
        time_correction = (self._master_clock() -
                           self._time_correction_fxns[clock_type]())
        if clock_type not in self._time_corrections:
            self._time_corrections[clock_type] = time_correction

        diff = time_correction - self._time_corrections[clock_type]
        max_dt = self._time_correction_maxs.get(clock_type, 10e-6)
        if np.abs(diff) > max_dt:
            logger.warning('Expyfun: drift of > {} microseconds ({}) '
                           'between {} clock and EC master clock.'
                           ''.format(max_dt * 1e6, int(round(diff * 1e6)),
                                     clock_type))
        logger.debug('Expyfun: time correction between {} clock and EC '
                     'master clock is {}. This is a change of {}.'
                     ''.format(clock_type, time_correction, time_correction -
                               self._time_corrections[clock_type]))
        return time_correction

    def wait_secs(self, secs):
        """Wait a specified number of seconds.

        Parameters
        ----------
        secs : float
            Number of seconds to wait.

        See Also
        --------
        ExperimentController.wait_until
        wait_secs
        """
        wait_secs(secs, ec=self)

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
            The difference between ``timestamp`` and the time `wait_until`
            was called.

        See Also
        --------
        ExperimentController.wait_secs
        wait_secs

        Notes
        -----
        Unlike `wait_secs`, there is no guarantee of precise timing with this
        function. It is the responsibility of the user to do choose a
        reasonable timestamp (or equivalently, do a reasonably small amount of
        processing prior to calling `wait_until`).
        """
        time_left = timestamp - self._master_clock()
        if time_left < 0:
            logger.warning('Expyfun: wait_until was called with a timestamp '
                           '({}) that had already passed {} seconds prior.'
                           ''.format(timestamp, -time_left))
        else:
            wait_secs(time_left, self)
        return time_left

    def identify_trial(self, **ids):
        """Identify trial type before beginning the trial

        Parameters
        ----------
        **ids : keyword arguments
            Ids to stamp, e.g. ``ec_id='TL90,MR45'. Use `id_types`
            to see valid options. Typical choices are ``ec_id``, ``el_id``,
            and ``ttl_id`` for experiment controller, eyelink, and TDT
            (or parallel port) respectively. If the value passed is a ``dict``,
            its entries will be passed as keywords to the underlying function.

        See Also
        --------
        ExperimentController.id_types
        ExperimentController.stamp_triggers
        ExperimentController.start_stimulus
        ExperimentController.stop
        ExperimentController.trial_ok
        """
        if self._trial_progress != 'stopped':
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
            if isinstance(id_, dict):
                self._id_call_dict[key](**id_)
            else:
                self._id_call_dict[key](id_)
        self._trial_progress = 'identified'

    def trial_ok(self):
        """Report that the trial was okay and do post-trial tasks.

        For example, logs and data files can be flushed at the end of each
        trial.

        See Also
        --------
        ExperimentController.identify_trial
        ExperimentController.start_stimulus
        ExperimentController.stop
        """
        if self._trial_progress != 'started':
            raise RuntimeError('trial cannot be okay unless it was started, '
                               'did you call ec.start_stimulus?')
        if self._playing:
            msg = 'ec.trial_ok called before stimulus had stopped'
            logger.warn(msg)
        for func in self._on_trial_ok:
            func()
        logger.exp('Expyfun: Trial OK')
        self._trial_progress = 'stopped'

    def _stamp_ec_id(self, id_):
        """Stamp id -- currently anything allowed"""
        self.write_data_line('trial_id', id_)

    def _stamp_binary_id(self, id_, delay=0.03, wait_for_last=True):
        """Helper for ec to stamp a set of IDs using binary controller

        This makes TDT and parallel port give the same output. Eventually
        we may want to customize it so that parallel could work differently,
        but for now it's unified. ``delay`` is the inter-trigger delay.
        """
        if not isinstance(id_, (list, tuple, np.ndarray)):
            raise TypeError('id must be array-like')
        id_ = np.array(id_)
        if not np.all(np.logical_or(id_ == 1, id_ == 0)):
            raise ValueError('All values of id must be 0 or 1')
        id_ = 2 ** (id_.astype(int) + 2)  # 4's and 8's
        # Note: we no longer put 8, 8 on ends
        self._stamp_ttl_triggers(id_, delay=delay, wait_for_last=wait_for_last)

    def stamp_triggers(self, ids, check='binary', wait_for_last=True):
        """Stamp binary values

        Parameters
        ----------
        ids : int | list of int
            Value(s) to stamp.
        check : str
            If 'binary', enforce standard binary value stamping of only values
            ``[1, 2, 4, 8]``. If 'int4', enforce values as integers between
            1 and 15.
        wait_for_last : bool
            If True, wait for last trigger to be stamped before returning.

        Notes
        -----
        This may be (nearly) instantaneous, or take a while, depending
        on the type of triggering (TDT or parallel).

        If absolute minimal latency is required, consider using the
        private function _stamp_ttl_triggers (for advanced use only,
        subject to change!).

        See Also
        --------
        ExperimentController.identify_trial
        """
        if check not in ('int4', 'binary'):
            raise ValueError('Check must be either "int4" or "binary"')
        ids = [ids] if not isinstance(ids, list) else ids
        if not all(isinstance(id_, int) and 1 <= id_ <= 15 for id_ in ids):
            raise ValueError('ids must all be integers between 1 and 15')
        if check == 'binary':
            _vals = [1, 2, 4, 8]
            if not all(id_ in _vals for id_ in ids):
                raise ValueError('with check="binary", ids must all be '
                                 '1, 2, 4, or 8: {0}'.format(ids))
        self._stamp_ttl_triggers(ids, wait_for_last=wait_for_last)

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
            cleanup_actions = [self._win.close] + cleanup_actions
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

# ############################## READ-ONLY PROPERTIES #########################
    @property
    def id_types(self):
        """Trial ID types needed for each trial.
        """
        return sorted(self._id_call_dict.keys())

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
    for key, val in d.items():
        if key in fixed:
            print('{0}: {1}'.format(key, val))
        else:
            d[key] = get_keyboard_input('{0}: '.format(key))


def get_keyboard_input(prompt, default=None, out_type=str, valid=None):
    """Get keyboard input of a specific type

    Parameters
    ----------
    prompt : str
        Prompt to use.
    default : object | None
        If user enters nothing, this will be used. If None, the user
        will be repeatedly prompted until a valid response is found.
    out_type : type
        Type to coerce to. If coersion fails, the user will be prompted
        again.
    valid : list | None
        An iterable that contains all the allowable inputs. Keeps asking until
        it recceives a valid input. Does not check if None.

    Returns
    -------
    response : of type `out_type`
        The user response.
    """
    # TODO: Let valid be an iterable OR a function handle, such that you could
    # pass a lambda, e.g., that made sure a float was in a given range
    # TODO: add tests
    if not isinstance(out_type, type):
        raise TypeError('out_type must be a type')
    good = False
    while not good:
        response = input(prompt)
        if response == '' and default is not None:
            response = default
        try:
            response = out_type(response)
        except ValueError:
            pass
        else:
            if valid is None or response in valid:
                good = True
    assert isinstance(response, out_type)
    return response


def _get_dev_db(audio_controller):
    """Selects device-specific amplitude to ensure equivalence across devices.
    """
    # First try to get the level from the expyfun.json file.
    level = get_config('DB_OF_SINE_AT_1KHZ_1RMS')
    if level is None:
        level = dict(
            RM1=108.,  # approx w/ knob @ 12 o'clock (knob not detented)
            RP2=108.,
            RP2legacy=108.,
            RZ6=114.,
            pyglet=100.,  # TODO: this value not calibrated, system-dependent
            dummy=90.,  # only used for testing
        ).get(audio_controller, None)
    else:
        level = float(level)
    if level is None:
        logger.warning('Expyfun: Unknown audio controller: stim scaler may '
                       'not work correctly. You may want to remove your '
                       'headphones if this is the first run of your '
                       'experiment.')
        level = 90  # for untested TDT models
    return level
