"""Tools for controlling experiment execution"""

# Authors: Dan McCloy <drmccloy@uw.edu>
#          Eric Larson <larsoner@uw.edu>
#          Jasper van den Bosch <jasperb@uw.edu>
#
# License: BSD (3-clause)

import inspect
import json
import os
import string
import sys
import traceback as tb
import warnings
from collections import OrderedDict
from functools import partial
from os import path as op

import numpy as np

from ._git import __version__, assert_version
from ._input_controllers import CedrusBox, Joystick, Keyboard, Mouse
from ._sound_controllers import _AUTO_BACKENDS, SoundCardController, SoundPlayer
from ._tdt_controller import TDTController
from ._trigger_controllers import ParallelTrigger
from ._utils import (
    ZeroClock,
    _check_pyglet_version,
    _fix_audio_dims,
    _get_args,
    _get_display,
    _sanitize,
    _TempDir,
    _wait_secs,
    check_units,
    date_str,
    flush_logger,
    get_config,
    logger,
    running_rms,
    set_log_file,
    verbose_dec,
)
from .visual import Rectangle, Text, Video, _convert_color

# Note: ec._trial_progress has three values:
# 1. 'stopped', which ec.identify_trial turns into...
# 2. 'identified', which ec.start_stimulus turns into...
# 3. 'started', which ec.trial_ok turns into 'stopped'.

_SLOW_LIMIT = 10000000


class ExperimentController:
    """Interface for hardware control (audio, buttonbox, eye tracker, etc.)

    Parameters
    ----------
    exp_name : str
        Name of the experiment.
    audio_controller : str | dict | None
        If audio_controller is None, the type will be read from the system
        configuration file. If a string, can be 'sound_card' or 'tdt',
        and the remaining audio parameters will be read from the
        machine configuration file. If a dict, must include a key 'TYPE' that
        is one of the supported types; the dict can contain other parameters
        specific to the backend (see :class:`TDTController` and
        :class:`SoundCardController`).
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
    monitor : dict | None
        Monitor properties. If dict, must include keys
        SCREEN_WIDTH, SCREEN_DISTANCE, and SCREEN_SIZE_PIX.
        Generally this can be ``None`` if the width and distance have been
        set properly for the machine in use.
    trigger_controller : str | None
        If ``None``, the type will be read from the system configuration file.
        If a string, must be 'dummy', 'parallel', 'sound_card', or 'tdt'.
        By default the mode is 'dummy', since setting up the parallel port
        can be a pain. Can also be a dict with entries 'TYPE' ('parallel'),
        and 'TRIGGER_ADDRESS' (None).
    session : str | None
        If ``None``, a GUI will be used to acquire this information.
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
    safe_flipping : bool | None
        If False, do not use ``glFinish`` when flipping. This can restore
        60 Hz on Linux systems where 30 Hz framerates occur, but the timing
        is not necessarily guaranteed, as the `flip` may return before the
        stimulus has actually flipped (check with
        :ref:`sphx_glr_auto_examples_sync_sync_test.py`).
    n_channels : int
        The number of audio playback channels. Defaults to 2 (must be 2 if
        a TDT is used).
    trigger_duration : float
        The trigger duration to use (sec). Must be 0.01 for TDT.
    joystick : bool
        Whether or not to enable joystick control.
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
    def __init__(
        self,
        exp_name,
        audio_controller=None,
        response_device=None,
        stim_rms=0.01,
        stim_fs=24414,
        stim_db=65,
        noise_db=45,
        output_dir="data",
        window_size=None,
        screen_num=None,
        full_screen=True,
        force_quit=None,
        participant=None,
        monitor=None,
        trigger_controller=None,
        session=None,
        check_rms="windowed",
        suppress_resamp=False,
        version=None,
        safe_flipping=None,
        n_channels=2,
        trigger_duration=0.01,
        joystick=False,
        verbose=None,
    ):
        # initialize some values
        self._stim_fs = stim_fs
        self._stim_rms = stim_rms
        self._stim_db = stim_db
        self._noise_db = noise_db
        self._stim_scaler = None
        self._suppress_resamp = suppress_resamp
        self.video = None
        self._bgcolor = _convert_color("k")
        # placeholder for extra actions to do on flip-and-play
        self._on_every_flip = []
        self._on_next_flip = []
        self._on_trial_ok = []
        # placeholder for extra actions to run on close
        self._extra_cleanup_fun = []  # be aware of order when adding to this
        self._id_call_dict = dict(ec_id=self._stamp_ec_id)
        self._ac = None
        self._data_file = None
        self._clock = ZeroClock()
        self._master_clock = self._clock.get_time

        # put anything that could fail in this block to ensure proper cleanup!
        try:
            self._setup_event_loop()
            self.set_rms_checking(check_rms)
            # Check Pyglet version for safety
            _check_pyglet_version(raise_error=True)
            # assure proper formatting for force-quit keys
            if force_quit is None:
                force_quit = ["lctrl", "rctrl"]
            elif isinstance(force_quit, (int, str)):
                force_quit = [str(force_quit)]
            if "escape" in force_quit:
                logger.warning(
                    'Expyfun: using "escape" as a force-quit key '
                    "is not recommended because it has special "
                    "status in pyglet."
                )
            # check expyfun version
            if version is None:
                raise RuntimeError(
                    "You must specify an expyfun version string"
                    " to use ExperimentController, or specify "
                    "version='dev' to override."
                )
            elif version.lower() != "dev":
                assert_version(version)
            # set up timing
            # Use ZeroClock, which uses the "clock" fn but starts at zero
            self._time_corrections = dict()
            self._time_correction_fxns = dict()
            self._time_correction_maxs = dict()  # optional, defaults to 50e-6

            # dictionary for experiment metadata
            self._exp_info = OrderedDict()

            for name in _get_args(self.__init__):
                if name != "self":
                    self._exp_info[name] = locals()[name]
            self._exp_info["date"] = date_str()
            # skip verbose decorator frames
            self._exp_info["file"] = op.abspath(inspect.getfile(sys._getframe(3)))
            self._exp_info["version_used"] = __version__

            # session start dialog, if necessary
            show_list = ["exp_name", "date", "file", "participant", "session"]
            edit_list = ["participant", "session"]  # things editable in GUI
            for key in show_list:
                value = self._exp_info[key]
                if (
                    key in edit_list
                    and value is not None
                    and not isinstance(value, str)
                ):
                    raise TypeError(f"{value} must be string or None")
                if key in edit_list and value is None:
                    self._exp_info[key] = get_keyboard_input(f"{key}: ")
                else:
                    print(f"{key}: {value}")

            #
            # initialize log file
            #
            self._output_dir = None
            set_log_file(None)
            if output_dir is not None:
                output_dir = op.abspath(output_dir)
                if not op.isdir(output_dir):
                    os.mkdir(output_dir)
                basename = op.join(
                    output_dir,
                    "{}_{}".format(
                        self._exp_info["participant"], self._exp_info["date"]
                    ),
                )
                self._output_dir = basename
                self._log_file = self._output_dir + ".log"
                set_log_file(self._log_file)
                closer = partial(set_log_file, None)
                # initialize data file
                self._data_file = open(self._output_dir + ".tab", "a")
                self._extra_cleanup_fun.append(self.flush)  # flush
                self._extra_cleanup_fun.append(self._data_file.close)  # close
                self._extra_cleanup_fun.append(closer)  # un-set log file
                self._data_file.write("# " + json.dumps(self._exp_info) + "\n")
                self.write_data_line("event", "value", "timestamp")
            logger.info(
                "Expyfun: Using version %s (requested %s)" % (__version__, version)
            )

            #
            # set up monitor
            #
            if safe_flipping is None:
                safe_flipping = not (get_config("SAFE_FLIPPING", "").lower() == "false")
            if not safe_flipping:
                logger.warning(
                    "Expyfun: Unsafe flipping mode enabled, flip timing not guaranteed"
                )
            self.safe_flipping = safe_flipping

            if screen_num is None:
                screen_num = int(get_config("SCREEN_NUM", "0"))
            display = _get_display()
            screen = display.get_screens()[screen_num]
            if monitor is None:
                mon_size = [screen.width, screen.height]
                mon_size = ",".join([str(d) for d in mon_size])
                monitor = dict()
                width = float(get_config("SCREEN_WIDTH", "51.0"))
                dist = float(get_config("SCREEN_DISTANCE", "48.0"))
                monitor["SCREEN_WIDTH"] = width
                monitor["SCREEN_DISTANCE"] = dist
                mon_size = get_config("SCREEN_SIZE_PIX", mon_size).split(",")
                mon_size = [int(p) for p in mon_size]
                monitor["SCREEN_SIZE_PIX"] = mon_size
            if not isinstance(monitor, dict):
                raise TypeError("monitor must be a dict, got %r" % (monitor,))
            req_mon_keys = ["SCREEN_WIDTH", "SCREEN_DISTANCE", "SCREEN_SIZE_PIX"]
            missing_keys = [key for key in req_mon_keys if key not in monitor]
            if missing_keys:
                raise KeyError(f"monitor is missing required keys {missing_keys}")
            mon_size = monitor["SCREEN_SIZE_PIX"]
            monitor["SCREEN_DPI"] = monitor["SCREEN_SIZE_PIX"][0] / (
                monitor["SCREEN_WIDTH"] * 0.393701
            )
            monitor["SCREEN_HEIGHT"] = (
                monitor["SCREEN_WIDTH"]
                / float(monitor["SCREEN_SIZE_PIX"][0])
                * float(monitor["SCREEN_SIZE_PIX"][1])
            )
            self._monitor = monitor

            #
            # parse audio controller
            #
            if audio_controller is None:
                audio_controller = {
                    "TYPE": get_config("AUDIO_CONTROLLER", "sound_card")
                }
            elif isinstance(audio_controller, str):
                # old option, backward compat / shortcut
                if audio_controller in _AUTO_BACKENDS:
                    audio_controller = {
                        "TYPE": "sound_card",
                        "SOUND_CARD_BACKEND": audio_controller,
                    }
                else:
                    audio_controller = {"TYPE": audio_controller.lower()}
            elif not isinstance(audio_controller, dict):
                raise TypeError(
                    "audio_controller must be a str or dict, got "
                    "type %s" % (type(audio_controller),)
                )
            audio_type = audio_controller["TYPE"].lower()

            #
            # parse response device
            #
            if response_device is None:
                response_device = get_config("RESPONSE_DEVICE", "keyboard")
            if response_device not in ["keyboard", "tdt", "cedrus"]:
                raise ValueError(
                    'response_device must be "keyboard", "tdt", "cedrus", or None'
                )
            self._response_device = response_device

            #
            # Initialize devices
            #

            trigger_duration = float(trigger_duration)
            if not 0.001 < trigger_duration <= 0.02:  # probably an error
                raise ValueError(
                    "high_duration must be between 0.001 and "
                    "0.02 sec, got %s" % (trigger_duration,)
                )

            # Audio (and for TDT, potentially keyboard)
            if audio_type == "tdt":
                logger.info("Expyfun: Setting up TDT")
                if n_channels != 2:
                    raise ValueError(
                        "n_channels must be equal to 2 for the "
                        "TDT backend, got %s" % (n_channels,)
                    )
                if trigger_duration != 0.01:
                    raise ValueError(
                        "trigger_duration must be 0.01 for TDT, "
                        "got %s" % (trigger_duration,)
                    )
                self._ac = TDTController(audio_controller, ec=self)
                self.audio_type = self._ac.model
            elif audio_type == "sound_card":
                self._ac = SoundCardController(
                    audio_controller,
                    self.stim_fs,
                    n_channels,
                    trigger_duration=trigger_duration,
                    ec=self,
                )
                self.audio_type = self._ac.backend_name
            else:
                raise ValueError(
                    "audio_controller['TYPE'] must be \"tdt\" "
                    'or "sound_card", got %r.' % (audio_type,)
                )
            del audio_type
            self._extra_cleanup_fun.insert(0, self._ac.halt)
            # audio scaling factor; ensure uniform intensity across devices
            self.set_stim_db(self._stim_db)
            self.set_noise_db(self._noise_db)

            if self._fs_mismatch:
                msg = (
                    "Expyfun: Mismatch between reported stim sample "
                    f"rate ({self.stim_fs}) and device sample rate ({self.fs}). "
                )
                if self._suppress_resamp:
                    msg += (
                        "Nothing will be done about this because "
                        'suppress_resamp is "True"'
                    )
                else:
                    msg += (
                        "Experiment Controller will resample for you, but "
                        "this takes a non-trivial amount of processing "
                        "time and may compromise your experimental "
                        "timing and/or cause artifacts."
                    )
                logger.warning(msg)

            #
            # set up visual window (must be done before keyboard and mouse)
            #
            logger.info("Expyfun: Setting up screen")
            if full_screen:
                if window_size is None:
                    window_size = monitor["SCREEN_SIZE_PIX"]
            else:
                if window_size is None:
                    window_size = get_config("WINDOW_SIZE", "800,600").split(",")
                    window_size = [int(w) for w in window_size]
            window_size = np.array(window_size)
            if window_size.ndim != 1 or window_size.size != 2:
                raise ValueError("window_size must be 2-element array-like or None")

            # open window and setup GL config
            self._setup_window(window_size, exp_name, full_screen, screen)

            # Keyboard
            if response_device == "keyboard":
                self._response_handler = Keyboard(self, force_quit)
            elif response_device == "tdt":
                if not isinstance(self._ac, TDTController):
                    raise ValueError(
                        'response_device can only be "tdt" if tdt is used for audio'
                    )
                self._response_handler = self._ac
                self._ac._add_keyboard_init(self, force_quit)
            else:  # response_device == 'cedrus'
                self._response_handler = CedrusBox(self, force_quit)

            # Joystick
            if joystick:
                self._joystick_handler = Joystick(self)
                self._extra_cleanup_fun.append(self._joystick_handler._close)
            else:
                self._joystick_handler = None

            #
            # set up trigger controller
            #
            self._ofp_critical_funs = list()
            if trigger_controller is None:
                trigger_controller = get_config("TRIGGER_CONTROLLER", "dummy")
            if isinstance(trigger_controller, str):
                trigger_controller = dict(TYPE=trigger_controller)
            assert isinstance(trigger_controller, dict)
            trigger_controller = trigger_controller.copy()
            known_keys = ("TYPE",)
            if set(trigger_controller) != set(known_keys):
                raise ValueError(
                    "Unknown keys for trigger_controller, must be "
                    f"{known_keys}, got {set(trigger_controller)}"
                )
            logger.info(
                f"Expyfun: Initializing {trigger_controller['TYPE']} triggering mode"
            )
            if trigger_controller["TYPE"] == "tdt":
                if not isinstance(self._ac, TDTController):
                    raise ValueError(
                        'trigger_controller can only be "tdt" if tdt is used for audio'
                    )
                self._tc = self._ac
            elif trigger_controller["TYPE"] == "sound_card":
                if not isinstance(self._ac, SoundCardController):
                    raise ValueError(
                        "trigger_controller can only be "
                        '"sound_card" if the sound card is '
                        "used for audio"
                    )
                if self._ac._n_channels_stim == 0:
                    raise ValueError(
                        "cannot use sound card for triggering "
                        "when SOUND_CARD_TRIGGER_CHANNELS is "
                        "zero"
                    )
                self._tc = self._ac
            elif trigger_controller["TYPE"] in ["parallel", "dummy"]:
                addr = trigger_controller.get(
                    "TRIGGER_ADDRESS", get_config("TRIGGER_ADDRESS", None)
                )
                self._tc = ParallelTrigger(
                    trigger_controller["TYPE"], addr, trigger_duration, ec=self
                )
                self._extra_cleanup_fun.insert(0, self._tc.close)
                # The TDT always stamps "1" on stimulus onset. Here we need
                # to manually mimic that behavior.
                self._ofp_critical_funs.insert(
                    0, lambda: self._stamp_ttl_triggers([1], False, False)
                )
            else:
                raise ValueError(
                    "trigger_controller type must be "
                    '"parallel", "dummy", "sound_card", or "tdt",'
                    "got {0}".format(trigger_controller["TYPE"])
                )
            self._id_call_dict["ttl_id"] = self._stamp_binary_id

            # other basic components
            self._mouse_handler = Mouse(self)
            t = np.arange(44100 // 3) / 44100.0
            car = sum([np.sin(2 * np.pi * f * t) for f in [800, 1000, 1200]])
            self._beep = None
            self._beep_data = np.tile(car * np.exp(-t * 10) / 4, (2, 3))

            # finish initialization
            logger.info("Expyfun: Initialization complete")
            logger.exp(
                "Expyfun: Participant: {0}".format(self._exp_info["participant"])
            )
            logger.exp("Expyfun: Session: {0}".format(self._exp_info["session"]))
            ok_log = partial(self.write_data_line, "trial_ok", None)
            self._on_trial_ok.append(ok_log)
            self._on_trial_ok.append(self.flush)
            self._trial_progress = "stopped"
        except Exception:
            self.close()
            raise
        # hack to prevent extra flips on first screen_prompt / screen_text
        self.flip()

    def __repr__(self):
        """Return a useful string representation of the experiment"""
        string = '<ExperimentController ({3}): "{0}" {1} ({2})>'.format(
            self._exp_info["exp_name"],
            self._exp_info["participant"],
            self._exp_info["session"],
            self.audio_type,
        )
        return string

    # ############################### SCREEN METHODS ##############################
    def screen_text(
        self,
        text,
        pos=(0, 0),
        color="white",
        font_name="Arial",
        font_size=24,
        wrap=True,
        units="norm",
        attr=True,
        log_data=True,
    ):
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
        log_data : bool
            Whether or not to write a line in the log file.

        Returns
        -------
        Instance of visual.Text

        See Also
        --------
        ExperimentController.screen_prompt
        """
        check_units(units)
        scr_txt = Text(
            self,
            text,
            pos,
            color,
            font_name,
            font_size,
            wrap=wrap,
            units=units,
            attr=attr,
        )
        scr_txt.draw()
        if log_data:
            self.call_on_next_flip(partial(self.write_data_line, "screen_text", text))
        return scr_txt

    def screen_prompt(
        self,
        text,
        max_wait=np.inf,
        min_wait=0,
        live_keys=None,
        timestamp=False,
        clear_after=True,
        pos=(0, 0),
        color="white",
        font_name="Arial",
        font_size=24,
        wrap=True,
        units="norm",
        attr=True,
        click=False,
    ):
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
        timestamp : bool
            If True, output the timestamp as well.
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
        click : bool
            Whether to use clicks to advance rather than key presses.

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
        if not all(isinstance(t, str) for t in text):
            raise TypeError("text must be a string or list of strings")
        for t in text:
            self.screen_text(
                t,
                pos=pos,
                color=color,
                font_name=font_name,
                font_size=font_size,
                wrap=wrap,
                units=units,
                attr=attr,
            )
            self.flip()
            fun = self.wait_one_click if click else self.wait_one_press
            out = fun(max_wait, min_wait, live_keys, timestamp)
        if clear_after:
            self.flip()
        return out

    def set_background_color(self, color="black"):
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
        appropriate background color.
        """
        from pyglet import gl

        # we go a little over here to be safe from round-off errors
        Rectangle(self, pos=[0, 0, 2.1, 2.1], fill_color=color).draw()
        self._bgcolor = _convert_color(color)
        gl.glClearColor(*[c / 255.0 for c in self._bgcolor])

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
            if self._trial_progress != "identified":
                raise RuntimeError("Trial ID must be stamped before starting the trial")
            self._trial_progress = "started"
        extra = "flipping screen and " if flip else ""
        logger.exp(f"Expyfun: Starting stimuli: {extra}playing audio")
        # ensure self._play comes first in list, followed by other critical
        # private functions (e.g., EL stamping), then user functions:
        if flip:
            self._on_next_flip = (
                [self._play] + self._ofp_critical_funs + self._on_next_flip
            )
            stimulus_time = self.flip(when)
        else:
            if when is not None:
                self.wait_until(when)
            funs = [self._play] + self._ofp_critical_funs
            self._dispatch_events()
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
        called with arguments, use :func:`functools.partial` before passing
        to `call_on_next_flip`.
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
        called with arguments, use :func:`functools.partial` before passing to
        `call_on_every_flip`.
        """
        if function is not None:
            self._on_every_flip.append(function)
        else:
            self._on_every_flip = []

    def _convert_units(self, verts, fro, to):
        """Convert between different screen units."""
        check_units(to)
        check_units(fro)
        verts = np.array(np.atleast_2d(verts), dtype=float)
        if verts.shape[0] != 2:
            raise RuntimeError("verts must have 2 rows")

        if fro == to:
            return verts

        # simplify by using two if neither is in normalized (native) units
        if "norm" not in [to, fro]:
            # convert to normal
            verts = self._convert_units(verts, fro, "norm")
            # convert from normal to dest
            verts = self._convert_units(verts, "norm", to)
            return verts

        # figure out our actual transition, knowing one is 'norm'
        win_w_pix, win_h_pix = self.window_size_pix
        mon_w_pix, mon_h_pix = self.monitor_size_pix
        wh_cm = np.array(
            [self._monitor["SCREEN_WIDTH"], self._monitor["SCREEN_HEIGHT"]], float
        )
        d_cm = self._monitor["SCREEN_DISTANCE"]
        cm_factors = (self.window_size_pix / self.monitor_size_pix * wh_cm / 2.0)[
            :, np.newaxis
        ]
        if "pix" in [to, fro]:
            if "pix" == to:
                # norm to pixels
                x = np.array(
                    [
                        [win_w_pix / 2.0, 0, win_w_pix / 2.0],
                        [0, win_h_pix / 2.0, win_h_pix / 2.0],
                    ]
                )
            else:
                # pixels to norm
                x = np.array([[2.0 / win_w_pix, 0, -1.0], [0, 2.0 / win_h_pix, -1.0]])
            verts = np.dot(x, np.r_[verts, np.ones((1, verts.shape[1]))])
        elif "deg" in [to, fro]:
            if "deg" == to:
                # norm (window) to norm (whole screen), then to deg
                verts = np.rad2deg(np.arctan2(verts * cm_factors, d_cm))
            else:
                # deg to norm (whole screen), to norm (window)
                verts = (d_cm * np.tan(np.deg2rad(verts))) / cm_factors
        elif "cm" in [to, fro]:
            if "cm" == to:
                verts = verts * cm_factors
            else:
                verts = verts / cm_factors
        else:
            raise KeyError(f'unknown conversion "{fro}" to "{to}"')
        return verts

    def screenshot(self):
        """Capture the current displayed buffer

        This method must be called *before* flipping, because it captures
        the back buffer.

        Returns
        -------
        data : array, shape (h, w, 4)
            Screen pixel colors.
        """
        import pyglet
        from PIL import Image

        tempdir = _TempDir()
        fname = op.join(tempdir, "tmp.png")
        with open(fname, "wb") as fid:
            pyglet.image.get_buffer_manager().get_color_buffer().save(file=fid)
        with Image.open(fname) as img:
            data = np.array(img)
        del tempdir
        assert data.ndim == 3 and data.shape[-1] == 4
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
        return self._monitor["SCREEN_DPI"]

    @property
    def window_size_pix(self):
        return np.array([self._win.width, self._win.height])

    @property
    def monitor_size_pix(self):
        return np.array(self._monitor["SCREEN_SIZE_PIX"])

    # ############################### VIDEO METHODS ###############################
    def load_video(self, file_name, pos=(0, 0), units="norm", center=True):
        """Load a video.

        Parameters
        ----------
        file_name : str
            The filename.
        pos : tuple
            The screen position.
        units : str
            Units for `pos`. See `check_units` for options.
        center : bool
            If True, center the video.
        """
        try:
            from pyglet.media.exceptions import MediaFormatException
        except ImportError:  # < 1.4
            from pyglet.media import MediaFormatException
        try:
            self.video = Video(self, file_name, pos, units)
        except MediaFormatException as exp:
            raise RuntimeError(
                "Something is wrong; probably you tried to load a "
                "compressed video file but you do not have FFmpeg/Avbin "
                "installed. Download and install it; if you are on Windows, "
                "you may also need to manually copy the .dll file(s) "
                "from C:\\Windows\\system32 to C:\\Windows\\SysWOW64.:\n%s" % (exp,)
            )

    def delete_video(self):
        """Delete the video."""
        self.video._delete()
        self.video = None

    # ############################### PYGLET EVENTS ###############################
    # https://pyglet.readthedocs.io/en/latest/programming_guide/eventloop.html#dispatching-events-manually  # noqa

    def _setup_event_loop(self):
        from pyglet.app import event_loop, platform_event_loop

        event_loop.has_exit = False
        platform_event_loop.start()
        event_loop.dispatch_event("on_enter")
        event_loop.is_running = True
        self._extra_cleanup_fun.append(self._end_event_loop)
        # This is when Pyglet calls:
        #     ev._run()
        # which is a while loop with the contents of our dispatch_events.

    def _dispatch_events(self):
        import pyglet

        pyglet.clock.tick()
        self._win.dispatch_events()
        # timeout = self._event_loop.idle()
        timeout = 0
        pyglet.app.platform_event_loop.step(timeout)

    def _end_event_loop(self):
        from pyglet.app import event_loop, platform_event_loop

        event_loop.is_running = False
        event_loop.dispatch_event("on_exit")
        platform_event_loop.stop()

    # ############################### OPENGL METHODS ##############################
    def _setup_window(self, window_size, exp_name, full_screen, screen):
        import pyglet
        from pyglet import gl

        # Use 16x sampling here
        config_kwargs = dict(
            depth_size=8,
            double_buffer=True,
            stereo=False,
            stencil_size=0,
            samples=0,
            sample_buffers=0,
        )
        # Travis can't handle multi-sampling, but our production machines must
        if os.getenv("TRAVIS") == "true":
            del config_kwargs["samples"], config_kwargs["sample_buffers"]
        self._full_screen = full_screen
        win_kwargs = dict(
            width=int(window_size[0]),
            height=int(window_size[1]),
            caption=exp_name,
            fullscreen=False,
            screen=screen,
            style="borderless",
            visible=False,
            config=pyglet.gl.Config(**config_kwargs),
        )

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
            x = int(win.screen.width / 2.0 - win.width / 2.0) + screen.x
            y = int(win.screen.height / 2.0 - win.height / 2.0) + screen.y
            win.set_location(x, y)
        self._win = win
        # with the context set up, do basic GL initialization
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)  # set the color to clear to
        gl.glClearDepth(1.0)  # clear value for the depth buffer
        # set the viewport size
        gl.glViewport(0, 0, int(self.window_size_pix[0]), int(self.window_size_pix[1]))
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
        v_ = False if os.getenv("_EXPYFUN_WIN_INVISIBLE") == "true" else True
        self.set_visible(v_)  # this is when we set fullscreen
        # ensure we got the correct window size
        got_size = win.get_size()
        if not np.array_equal(got_size, window_size):
            raise RuntimeError(
                "Window size requested by config (%s) does not "
                "match obtained window size (%s), is the "
                "screen resolution set incorrectly?" % (window_size, got_size)
            )
        self._dispatch_events()
        logger.info(
            "Initialized %s window on screen %s with DPI %0.2f"
            % (window_size, screen, self.dpi)
        )

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
        self._dispatch_events()
        self._win.switch_to()
        if self.safe_flipping:
            # On NVIDIA Linux these calls cause a 2x delay (33ms instead of 16)
            gl.glFinish()
        self._win.flip()
        # this waits until everything is called, including last draw
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glBegin(gl.GL_POINTS)
        gl.glColor4f(0, 0, 0, 0)
        gl.glVertex2i(10, 10)
        gl.glEnd()
        if self.safe_flipping:
            gl.glFinish()
        flip_time = self.get_time()
        for function in call_list:
            function()
        self.write_data_line("flip", flip_time)
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
        return 1.0 / np.median(np.diff(times[1:]))

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
        self._win.set_fullscreen(self._full_screen)
        self._win.set_visible(visible)
        logger.exp(f"Expyfun: Set screen visibility {visible}")
        if visible and flip:
            self.flip()
            # it seems like newer Pyglet sometimes messes up without two flips
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

    def get_presses(
        self,
        live_keys=None,
        timestamp=True,
        relative_to=None,
        kind="presses",
        return_kinds=False,
    ):
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
            live_keys, timestamp, relative_to, kind, return_kinds
        )

    def wait_one_press(
        self,
        max_wait=np.inf,
        min_wait=0.0,
        live_keys=None,
        timestamp=True,
        relative_to=None,
    ):
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
            max_wait, min_wait, live_keys, timestamp, relative_to
        )

    def wait_for_presses(
        self, max_wait, min_wait=0.0, live_keys=None, timestamp=True, relative_to=None
    ):
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
            max_wait, min_wait, live_keys, timestamp, relative_to
        )

    def _log_presses(self, pressed, kind="key"):
        """Write key presses to data file."""
        # This function will typically be called by self._response_handler
        # after it retrieves some button presses
        for key, stamp, eventType in pressed:
            self.write_data_line(kind + eventType, key, stamp)

    def check_force_quit(self):
        """Check to see if any force quit keys were pressed."""
        self._response_handler.check_force_quit()

    def text_input(
        self,
        stop_key="return",
        instruction_string="Type response below",
        pos=(0, 0),
        color="white",
        font_name="Arial",
        font_size=24,
        wrap=True,
        units="norm",
        all_caps=True,
    ):
        """Allows participant to input text and view on the screen.

        Parameters
        ----------
        stop_key : str
            The key to exit text input mode.
        instruction_string : str
            A string after the text entry to tell the participant what to do.
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
        all_caps : bool
            Whether the text should be displayed in all caps.

        Returns
        -------
        text : str
            The final input string.
        """
        letters = string.ascii_letters + " "
        text = ""
        while True:
            self.screen_text(
                instruction_string + "\n\n" + text + "|",
                pos=pos,
                color=color,
                font_name=font_name,
                font_size=font_size,
                wrap=wrap,
                units=units,
                log_data=False,
            )
            self.flip()
            letter = self.wait_one_press(timestamp=False)
            if letter == stop_key:
                self.flip()
                break
            if letter == "backspace":
                text = text[:-1]
            else:
                letter = " " if letter == "space" else letter
                letter = letter.upper() if all_caps else letter
                text += letter if letter in letters else ""
        self.write_data_line("text_input", text)
        return text

    # ############################## KEYPRESS METHODS #############################
    def listen_joystick_button_presses(self):
        """Start listening for joystick buttons.

        See Also
        --------
        ExperimentController.get_joystick_button_presses
        """
        self._joystick_handler.listen_presses()

    def get_joystick_button_presses(
        self, timestamp=True, relative_to=None, kind="presses", return_kinds=False
    ):
        """Get the entire joystick buffer.

        This will also clear events that are not requested per ``type``.

        Parameters
        ----------
        timestamp : bool
            Whether the keypress should be timestamped. If True, returns the
            button press time relative to the value given in `relative_to`.
        relative_to : None | float
            A time relative to which timestamping is done. Ignored if
            timestamp==False.  If ``None``, timestamps are relative to the time
            `listen_presses` was last called.
        kind : string
            Which button events to return. One of ``presses``, ``releases`` or
            ``both``. (default ``presses``)
        return_kinds : bool
            Return the kinds of presses.

        Returns
        -------
        presses : list
            Returns a list of tuples with button events. Each tuple's first
            value will be the button pressed. If ``timestamp==True``, the
            second value is the time for the event. If ``return_kinds==True``,
            then the last value is a string indicating if this was a button
            press or release event.

        See Also
        --------
        ExperimentController.listen_presses
        """
        self._dispatch_events()
        return self._joystick_handler.get_presses(
            None, timestamp, relative_to, kind, return_kinds
        )

    def get_joystick_value(self, kind):
        """Get the current joystick x direction.

        Parameters
        ----------
        kind : str
            Can be "x", "y", "hat_x", "hat_y", "z", "rz", "rx", or "ry".

        Returns
        -------
        x : float
            Value in the range -1 to 1.
        """
        return getattr(self._joystick_handler, kind)

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
        return self._mouse_handler.get_clicks(live_buttons, timestamp, relative_to)

    def get_mouse_position(self, units="pix"):
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
        pos = self._convert_units(pos[:, np.newaxis], "norm", units)[:, 0]
        return pos

    def toggle_cursor(self, visibility, flip=False):
        """Show or hide the mouse

        Parameters
        ----------
        visibility : bool
            If True, show; if False, hide.
        flip : bool
            If True, flip after toggling.

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

    def wait_one_click(
        self,
        max_wait=np.inf,
        min_wait=0.0,
        live_buttons=None,
        timestamp=True,
        relative_to=None,
        visible=None,
    ):
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
        return self._mouse_handler.wait_one_click(
            max_wait, min_wait, live_buttons, timestamp, relative_to, visible
        )

    def move_mouse_to(self, pos, units="norm"):
        """Move the mouse position to the specified position.

        Parameters
        ----------
        pos : array-like
            2-element array-like with X and Y.
        units : str
            Units to use. See ``check_units`` for options.
        """
        self._mouse_handler._move_to(pos, units)

    def wait_for_clicks(
        self,
        max_wait=np.inf,
        min_wait=0.0,
        live_buttons=None,
        timestamp=True,
        relative_to=None,
        visible=None,
    ):
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
        return self._mouse_handler.wait_for_clicks(
            max_wait, min_wait, live_buttons, timestamp, relative_to, visible
        )

    def wait_for_click_on(
        self,
        objects,
        max_wait=np.inf,
        min_wait=0.0,
        live_buttons=None,
        timestamp=True,
        relative_to=None,
    ):
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
            raise TypeError("objects must be a list or one of: %s" % (legal_types,))
        return self._mouse_handler.wait_for_click_on(
            objects, max_wait, min_wait, live_buttons, timestamp, relative_to
        )

    def _log_clicks(self, clicked):
        """Write mouse clicks to data file."""
        # This function will typically be called by self._response_handler
        # after it retrieves some mouse clicks
        for button, x, y, stamp in clicked:
            self.write_data_line("mouseclick", "%s,%i,%i" % (button, x, y), stamp)

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
        ExperimentController.set_noise_db
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
            raise RuntimeError(
                "Previous audio must be stopped before loading the buffer"
            )
        samples = self._validate_audio(samples)
        if not np.isclose(self._stim_scaler, 1.0):
            samples = samples * self._stim_scaler
        logger.exp(f"Expyfun: Loading {samples.size} samples to buffer")
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
        logger.exp("Expyfun: Playing audio")
        # ensure self._play comes first in list:
        self._play()
        return self.get_time()

    def _play(self):
        """Play the audio buffer."""
        if self._playing:
            raise RuntimeError("Previous audio must be stopped before playing")
        self._ac.play()
        logger.debug("Expyfun: started audio")
        self.write_data_line("play")

    @property
    def _playing(self):
        """Whether or not a stimulus is currently playing"""
        return self._ac.playing

    def stop(self, wait=False):
        """Stop audio buffer playback and reset cursor to beginning of buffer

        Parameters
        ----------
        wait : bool
            If True, try to wait until the end of the sound stimulus
            (not guaranteed to yield accurate timings!).

        See Also
        --------
        ExperimentController.load_buffer
        ExperimentController.play
        ExperimentController.set_stim_db
        ExperimentController.start_stimulus
        """
        if self._ac is not None:  # need to check b/c used in __exit__
            self._ac.stop(wait=wait)
        self.write_data_line("stop")
        logger.exp("Expyfun: Audio stopped and reset.")

    def set_noise_db(self, new_db):
        """Set the level of the background noise.

        Parameters
        ----------
        new_db : float
            The new level.

        See Also
        --------
        ExperimentController.start_noise
        ExperimentController.stop_noise
        """
        # Noise is always generated at an RMS of 1
        self._ac.set_noise_level(self._update_sound_scaler(new_db, 1.0))
        self._noise_db = new_db

    def set_stim_db(self, new_db):
        """Set the level of the stimuli.

        Parameters
        ----------
        new_db : float
            The new level.

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
        """Calcs coefficient ensuring stim ampl equivalence across devices."""
        exponent = -(_get_dev_db(self.audio_type) - desired_db) / 20.0
        return (10**exponent) / float(orig_rms)

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
        if samples.size and np.max(np.abs(samples)) > 1:
            raise ValueError("Sound data exceeds +/- 1.")
            # samples /= np.max(np.abs(samples),axis=0)

        # check shape and dimensions, make stereo
        samples = _fix_audio_dims(samples, self._ac._n_channels).T

        # This limit is currently set by the TDT SerialBuf objects
        # (per channel), it sets the limit on our stimulus durations...
        if np.isclose(self.stim_fs, 24414, atol=1):
            max_samples = 4000000 - 1
            if samples.shape[0] > max_samples:
                raise RuntimeError(
                    f"Sample too long {samples.shape[0]} > {max_samples}"
                )

        # resample if needed
        if self._fs_mismatch and not self._suppress_resamp:
            logger.warning(
                f"Expyfun: Resampling {round(len(samples) / self.stim_fs, 2)} "
                "seconds of audio"
            )
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("ignore")
                from mne.filter import resample
            if samples.size:
                samples = resample(
                    samples.astype(np.float64), self.fs, self.stim_fs, axis=0
                ).astype(np.float32)

        # check RMS
        if self._check_rms is not None and samples.size:
            chans = [samples[:, x] for x in range(samples.shape[1])]
            if self._check_rms == "wholefile":
                chan_rms = [np.sqrt(np.mean(x**2)) for x in chans]
                max_rms = max(chan_rms)
            else:  # 'windowed'
                # ~226 sec at 44100 Hz
                if samples.size >= _SLOW_LIMIT and not self._slow_rms_warned:
                    warnings.warn(
                        "Checking RMS with a 10 ms window and many samples is "
                        'slow, consider using None or "wholefile" modes.'
                    )
                    self._slow_rms_warned = True
                win_length = int(self.fs * 0.01)  # 10ms running window
                max_rms = [running_rms(x, win_length).max() for x in chans]
                max_rms = max(max_rms)
            if max_rms > 2 * self._stim_rms:
                warn_string = (
                    f"Expyfun: Stimulus max RMS ({max_rms}) exceeds stated "
                    f"RMS ({self._stim_rms}) by more than 6 dB."
                    ""
                )
                logger.warning(warn_string)
                warnings.warn(warn_string)
            elif max_rms < 0.5 * self._stim_rms:
                warn_string = (
                    f"Expyfun: Stimulus max RMS ({max_rms}) is less than "
                    f"stated RMS ({self._stim_rms}) by more than 6 dB."
                    ""
                )
                logger.warning(warn_string)

        # let's make sure we don't change our version of this array later
        samples = samples.view()
        samples.flags["WRITEABLE"] = False
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
        if check_rms not in [None, "wholefile", "windowed"]:
            raise ValueError(
                'check_rms must be one of "wholefile", "windowed", or None.'
            )
        self._slow_rms_warned = False
        self._check_rms = check_rms

    # ############################## OTHER METHODS ################################
    @property
    def participant(self):
        return self._exp_info["participant"]

    @property
    def session(self):
        return self._exp_info["session"]

    @property
    def exp_name(self):
        return self._exp_info["exp_name"]

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
        ll = "\t".join(_sanitize(x) for x in [timestamp, event_type, value]) + "\n"
        if self._data_file is not None:
            if self._data_file.closed:
                logger.warning(
                    "Data line not written due to closed file %s:\n"
                    "%s" % (self.data_fname, ll[:-1])
                )
            else:
                self._data_file.write(ll)
            self.flush()

    def _get_time_correction(self, clock_type):
        """Clock correction (sec) for different devices (screen, bbox, etc.)"""
        time_correction = (
            self._master_clock() - self._time_correction_fxns[clock_type]()
        )
        if clock_type not in self._time_corrections:
            self._time_corrections[clock_type] = time_correction

        diff = time_correction - self._time_corrections[clock_type]
        max_dt = self._time_correction_maxs.get(clock_type, 50e-6)
        if np.abs(diff) > max_dt:
            logger.warning(
                f"Expyfun: drift of > {max_dt * 1e6} microseconds "
                f"({int(round(diff * 1e6))}) "
                f"between {clock_type} clock and EC master clock."
            )
        logger.debug(
            f"Expyfun: time correction between {clock_type} clock and EC "
            f"master clock is {time_correction}. This is a change of "
            f"{time_correction - self._time_corrections[clock_type]}."
        )
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
        """
        # hog the cpu, checking time
        _wait_secs(secs, self)

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

        Notes
        -----
        Unlike `wait_secs`, there is no guarantee of precise timing with this
        function. It is the responsibility of the user to do choose a
        reasonable timestamp (or equivalently, do a reasonably small amount of
        processing prior to calling `wait_until`).
        """
        time_left = timestamp - self._master_clock()
        if time_left < 0:
            logger.warning(
                "Expyfun: wait_until was called with a timestamp "
                f"({timestamp}) that had already passed {-time_left} seconds prior."
                ""
            )
        else:
            self.wait_secs(time_left)
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
        if self._trial_progress != "stopped":
            raise RuntimeError("Cannot identify a trial twice")
        call_set = set(self._id_call_dict.keys())
        passed_set = set(ids.keys())
        if not call_set == passed_set:
            raise KeyError(
                f"All keys passed in {passed_set} must match the set of "
                f"keys required {call_set}"
            )
        ll = max([len(key) for key in ids.keys()])
        for key, id_ in ids.items():
            logger.exp(f"Expyfun: Stamp trial ID to {key.ljust(ll)} : {str(id_)}")
            if isinstance(id_, dict):
                self._id_call_dict[key](**id_)
            else:
                self._id_call_dict[key](id_)
        self._trial_progress = "identified"

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
        if self._trial_progress != "started":
            raise RuntimeError(
                "trial cannot be okay unless it was started, "
                "did you call ec.start_stimulus?"
            )
        if self._playing:
            logger.warning("ec.trial_ok called before stimulus had stopped")
        for func in self._on_trial_ok:
            func()
        logger.exp("Expyfun: Trial OK")
        self._trial_progress = "stopped"

    def _stamp_ec_id(self, id_):
        """Stamp id -- currently anything allowed"""
        self.write_data_line("trial_id", id_)

    def _stamp_binary_id(self, id_, wait_for_last=True):
        """Helper for ec to stamp a set of IDs using binary controller

        This makes TDT and parallel port give the same output. Eventually
        we may want to customize it so that parallel could work differently,
        but for now it's unified. ``delay`` is the inter-trigger delay.
        """
        if not isinstance(id_, (list, tuple, np.ndarray)):
            raise TypeError("id must be array-like")
        id_ = np.array(id_)
        if not np.all(np.isin(id_, [0, 1])):
            raise ValueError("All values of id must be 0 or 1")
        id_ = (id_.astype(int) + 1) << 2  # 0, 1 -> 4, 8
        self._stamp_ttl_triggers(id_, wait_for_last, True)

    def stamp_triggers(self, ids, check="binary", wait_for_last=True):
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
            If False, don't wait at all (if possible).

        Notes
        -----
        This may be (nearly) instantaneous, or take a while, depending
        on the type of triggering (TDT, sound card, or parallel).

        See Also
        --------
        ExperimentController.identify_trial
        """
        if check not in ("int4", "binary"):
            raise ValueError('Check must be either "int4" or "binary"')
        ids = [ids] if not isinstance(ids, list) else ids
        if not all(isinstance(id_, int) and 1 <= id_ <= 15 for id_ in ids):
            raise ValueError("ids must all be integers between 1 and 15")
        if check == "binary":
            _vals = [1, 2, 4, 8]
            if not all(id_ in _vals for id_ in ids):
                raise ValueError(
                    f'with check="binary", ids must all be 1, 2, 4, or 8: {ids}'
                )
        self._stamp_ttl_triggers(ids, wait_for_last, False)

    def _stamp_ttl_triggers(self, ids, wait_for_last, is_trial_id):
        logger.exp("Stamping TTL triggers: %s", ids)
        self._tc.stamp_triggers(
            ids, wait_for_last=wait_for_last, is_trial_id=is_trial_id
        )
        self.flush()

    def flush(self):
        """Flush logs and data files."""
        flush_logger()
        if self._data_file is not None and not self._data_file.closed:
            self._data_file.flush()

    def close(self):
        """Close all connections in experiment controller."""
        self.__exit__(None, None, None)

    def __enter__(self):
        logger.debug("Expyfun: Entering")
        return self

    def __exit__(self, err_type, value, traceback):
        """Exit cleanly.

        err_type, value and traceback will be None when called by self.close()
        """
        logger.info("Expyfun: Exiting")
        # do external cleanups
        cleanup_actions = []
        if hasattr(self, "_win"):
            cleanup_actions.append(self._win.close)
        cleanup_actions.extend([self.stop_noise, self.stop])
        cleanup_actions.extend(self._extra_cleanup_fun)
        cleanup_actions.append(self.flush)  # probably shouldn't be necessary
        for action in cleanup_actions:
            try:
                action()
            except Exception:
                tb.print_exc()
        if any([x is not None for x in (err_type, value, traceback)]):
            return False
        return True

    def refocus(self):
        """Attempt to grab operating system window manager / keyboard focus.

        This implements platform-specific trickery to bring the window to the
        top and capture keyboard focus in cases where keyboard input from a
        subject is mandatory (e.g., when using keyboard as a response device).

        Notes
        -----
        For Windows, the solution as adapted from:

            https://stackoverflow.com/questions/916259/win32-bring-a-window-to-top#answer-34414846

        This function currently does nothing on Linux and OSX.
        """  # noqa: E501
        if sys.platform == "win32":
            from pyglet.libs.win32 import _user32

            m_hWnd = self._win._hwnd
            hCurWnd = _user32.GetForegroundWindow()
            if hCurWnd != m_hWnd:
                # m_hWnd, HWND_TOPMOST, ..., SWP_NOSIZE | SWP_NOMOVE
                _user32.SetWindowPos(m_hWnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002)
                dwMyID = _user32.GetWindowThreadProcessId(m_hWnd, 0)
                dwCurID = _user32.GetWindowThreadProcessId(hCurWnd, 0)
                _user32.AttachThreadInput(dwCurID, dwMyID, True)
                self._win.activate()  # _user32.SetForegroundWindow(m_hWnd)
                _user32.AttachThreadInput(dwCurID, dwMyID, False)
                _user32.SetFocus(m_hWnd)
                _user32.SetActiveWindow(m_hWnd)

    # ############################## READ-ONLY PROPERTIES #########################
    @property
    def id_types(self):
        """Trial ID types needed for each trial."""
        return sorted(self._id_call_dict.keys())

    @property
    def fs(self):
        """Playback frequency of the audio controller (samples / second)."""
        return self._ac.fs  # not user-settable

    @property
    def stim_fs(self):
        """Sampling rate at which the stimuli were generated."""
        return self._stim_fs  # not user-settable

    @property
    def stim_db(self):
        """Sound power in dB of the stimuli."""
        return self._stim_db  # not user-settable

    @property
    def noise_db(self):
        """Sound power in dB of the background noise."""
        return self._noise_db  # not user-settable

    @property
    def current_time(self):
        """Timestamp from the experiment master clock."""
        return self._master_clock()

    @property
    def _fs_mismatch(self):
        """Quantify if sample rates substantively differ."""
        return not np.allclose(self.stim_fs, self.fs, rtol=0, atol=0.5)

    # Testing cruft to work around "queue full" errors on Windows
    def _ac_flush(self):
        if isinstance(getattr(self, "_ac", None), SoundCardController):
            self._ac.halt()


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
        Type to coerce to. If coercion fails, the user will be prompted
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
        raise TypeError("out_type must be a type")
    good = False
    while not good:
        response = input(prompt)
        if response == "" and default is not None:
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
    """Selects device-specific amplitude to ensure equivalence across devices."""
    # First try to get the level from the expyfun.json file.
    level = get_config("DB_OF_SINE_AT_1KHZ_1RMS")
    if level is None:
        level = dict(
            RM1=108.0,  # approx w/ knob @ 12 o'clock (knob not detented)
            RP2=108.0,
            RP2legacy=108.0,
            RZ6=114.0,
            # TODO: these values not calibrated, system-dependent
            pyglet=100.0,
            rtmixer=100.0,
            dummy=100.0,  # only used for testing
        ).get(audio_controller, None)
    else:
        level = float(level)
    if level is None:
        logger.warning(
            "Expyfun: Unknown audio controller %s: stim scaler may "
            "not work correctly. You may want to remove your "
            "headphones if this is the first run of your "
            "experiment." % (audio_controller,)
        )
        level = 100  # for untested TDT models
    return level
