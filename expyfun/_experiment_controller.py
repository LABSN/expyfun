"""Tools for controlling experiment execution"""

# Authors: Dan McCloy <drmccloy@uw.edu>
#          Eric Larson <larsoner@uw.edu>
#
# License: BSD (3-clause)

import numpy as np
import os
from os import path as op
from functools import partial
from scipy.signal import resample
from psychopy import visual, core, event, sound, gui, monitors, misc
from psychopy.data import getDateStr as date_str
from ._utils import (get_config, verbose_dec, _check_pyglet_version, wait_secs,
                     running_rms, _sanitize, psylog)
from ._tdt_controller import TDTController
from ._trigger_controllers import PsychTrigger
from ._sound_controllers import PsychSound
from ._input_controllers import PsychKeyboard


class ExperimentController(object):
    """Interface for hardware control (audio, buttonbox, eye tracker, etc.)

    Parameters
    ----------
    exp_name : str
        Name of the experiment.
    audio_controller : str | dict | None
        If audio_controller is None, the type will be read from the system
        configuration file. If a string, can be 'psychopy' or 'tdt', and the
        remaining audio parameters will be read from the machine configuration
        file. If a dict, must include a key 'TYPE' that is either 'psychopy'
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
                 verbose=None, check_rms='windowed'):

        # Check Pyglet version for safety
        _check_pyglet_version(raise_error=True)

        # initialize some values
        self._stim_fs = stim_fs
        self._stim_rms = stim_rms
        self._stim_db = stim_db
        self._noise_db = noise_db
        self._stim_scaler = None
        self.set_rms_checking(check_rms)
        # list of entities to draw / clear from the visual window
        self._screen_objects = []
        # placeholder for extra actions to do on flip-and-play
        self._on_every_flip = None
        self._on_next_flip = None
        # placeholder for extra actions to run on close
        self._extra_cleanup_fun = []
        # some hardcoded parameters...
        bkgd_color = [-1, -1, -1]  # psychopy does RGB from -1 to 1

        # assure proper formatting for force-quit keys
        if force_quit is None:
            force_quit = ['lctrl', 'rctrl']
        elif isinstance(force_quit, (int, basestring)):
            force_quit = [str(force_quit)]
        if 'escape' in force_quit:
            psylog.warn('Expyfun: using "escape" as a force-quit key is not '
                        'recommended because it has special status in pyglet.')

        # set up timing
        self._master_clock = core.MonotonicClock()

        # dictionary for experiment metadata
        self._exp_info = {'participant': participant, 'session': session,
                          'exp_name': exp_name, 'date': date_str()}

        # session start dialog, if necessary
        fixed_list = ['exp_name', 'date']  # things not user-editable in GUI
        for key, value in self._exp_info.iteritems():
            if key not in fixed_list and value is not None:
                if not isinstance(value, basestring):
                    raise TypeError('{} must be string or None'.format(value))
                fixed_list.append(key)

        if len(fixed_list) < len(self._exp_info):
            session_dialog = gui.DlgFromDict(dictionary=self._exp_info,
                                             fixed=fixed_list,
                                             title=exp_name)
            if not session_dialog.OK:
                self.close()  # user pressed cancel

        #
        # initialize log file
        #
        if output_dir is not None:
            output_dir = op.abspath(output_dir)
            if not op.isdir(output_dir):
                os.mkdir(output_dir)
            basename = op.join(output_dir,
                               '{}_{}'.format(self._exp_info['participant'],
                                              self._exp_info['date']))
            self._log_file = basename + '.log'
            psylog.LogFile(self._log_file, level=psylog.INFO)
            # initialize data file
            self._data_file = open(basename + '.tab', 'a')
            self._data_file.write('# ' + str(self._exp_info) + '\n')
            self.write_data_line('event', 'value', 'timestamp')
        else:
            psylog.LogFile(None, level=psylog.info)
            self._data_file = None

        #
        # set up monitor
        #
        if monitor is None:
            monitor = dict()
            monitor['SCREEN_WIDTH'] = float(get_config('SCREEN_WIDTH', '51.0'))
            monitor['SCREEN_DISTANCE'] = float(get_config('SCREEN_DISTANCE',
                                               '48.0'))
            mon_size = get_config('SCREEN_SIZE_PIX', '1920,1080').split(',')
            mon_size = [float(m) for m in mon_size]
            monitor['SCREEN_SIZE_PIX'] = mon_size
        else:
            if not isinstance(monitor, dict):
                raise TypeError('monitor must be a dict')
            if not all([key in monitor for key in ['SCREEN_WIDTH',
                                                   'SCREEN_DISTANCE',
                                                   'SCREEN_SIZE_PIX']]):
                raise KeyError('monitor must have keys "SCREEN_WIDTH", '
                               '"SCREEN_DISTANCE", and "SCREEN_SIZE_PIX"')
        mon_size = monitor['SCREEN_SIZE_PIX']
        monitor = monitors.Monitor('custom', monitor['SCREEN_WIDTH'],
                                   monitor['SCREEN_DISTANCE'])
        monitor.setSizePix(mon_size)

        #
        # parse audio controller
        #
        if audio_controller is None:
            audio_controller = {'TYPE': get_config('AUDIO_CONTROLLER',
                                                   'psychopy')}
        elif isinstance(audio_controller, basestring):
            if audio_controller.lower() in ['psychopy', 'tdt']:
                audio_controller = {'TYPE': audio_controller.lower()}
            else:
                raise ValueError('audio_controller must be \'psychopy\' or '
                                 '\'tdt\' (or a dict including \'TYPE\':'
                                 ' \'psychopy\' or \'TYPE\': \'tdt\').')
        elif not isinstance(audio_controller, dict):
            raise TypeError('audio_controller must be a str or dict.')
        self._audio_type = audio_controller['TYPE'].lower()

        #
        # parse response device
        #
        if response_device is None:
            response_device = get_config('RESPONSE_DEVICE', 'keyboard')
        if response_device not in ['keyboard', 'tdt']:
            raise ValueError('response_device must be "keyboard", "tdt", or '
                             'None')
        self._response_device = response_device

        #
        # Initialize devices
        #

        # Audio (and for TDT, potentially keyboard)
        self._tdt_init = False
        if self._audio_type == 'tdt':
            psylog.info('Expyfun: Setting up TDT')
            as_kb = True if self._response_device == 'tdt' else False
            self._ac = TDTController(audio_controller, self, as_kb, force_quit)
            self._audio_type = self._ac.model
            self._tdt_init = True
        elif self._audio_type == 'psychopy':
            psylog.info('Expyfun: Setting up PsychoPy audio with {} '
                        'backend'.format(sound.audioLib))
            self._ac = PsychSound(self, self.stim_fs)
        else:
            raise ValueError('audio_controller[\'TYPE\'] must be '
                             '\'psychopy\' or \'tdt\'.')
        # audio scaling factor; ensure uniform intensity across output devices
        self.set_stim_db(self._stim_db)
        self.set_noise_db(self._noise_db)

        if self._fs_mismatch is True:
            psylog.warn('Mismatch between reported stim sample rate ({0}) and '
                        'device sample rate ({1}). ExperimentController will '
                        'resample for you, but that takes a non-trivial amount'
                        ' of processing time and may compromise your '
                        'experimental timing and/or introduce artifacts.'
                        ''.format(self.stim_fs, self.fs))

        # Keyboard
        if response_device == 'keyboard':
            self._response_handler = PsychKeyboard(self, force_quit)
        if response_device == 'tdt':
            if self._tdt_init is False:
                raise ValueError('response_device can only be "tdt" if '
                                 'tdt is used for audio')
            self._response_handler = self._ac

        # pass on check force quit calls
        self._check_force_quit = self._response_handler.check_force_quit

        #
        # set up trigger controller
        #
        if trigger_controller is None:
            trigger_controller = get_config('TRIGGER_CONTROLLER', 'dummy')
        if isinstance(trigger_controller, basestring):
            trigger_controller = dict(type=trigger_controller)
        psylog.info('Initializing {} triggering mode'
                    ''.format(trigger_controller['type']))
        if trigger_controller['type'] == 'tdt':
            if self._tdt_init is False:
                raise ValueError('trigger_controller can only be "tdt" if '
                                 'tdt is used for audio')
            self._trigger_handler = self._ac
        elif trigger_controller['type'] in ['parallel', 'dummy']:
            if 'address' not in trigger_controller['type']:
                trigger_controller['address'] = get_config('TRIGGER_ADDRESS')
            out = PsychTrigger(trigger_controller['type'],
                               trigger_controller.get('address'))
            self._trigger_handler = out
            self._extra_cleanup_fun.append(self._trigger_handler.close)
        else:
            raise ValueError('trigger_controller type must be '
                             '"parallel", "dummy", or "tdt", not '
                             '{0}'.format(trigger_controller['type']))
        self._trigger_controller = trigger_controller['type']

        #
        # set up visual window
        #
        psylog.info('Expyfun: Setting up screen')
        if window_size is None:
            window_size = get_config('WINDOW_SIZE', '1920,1080').split(',')
        if screen_num is None:
            screen_num = int(get_config('SCREEN_NUM', '0'))
        self._win = visual.Window(size=window_size, fullscr=full_screen,
                                  monitor=monitor, screen=screen_num,
                                  winType='pyglet', allowGUI=False,
                                  allowStencil=False, color=bkgd_color,
                                  colorSpace='rgb')

        self._text_stim = visual.TextStim(win=self._win, text='', pos=[0, 0],
                                          height=0.1, wrapWidth=1.2,
                                          units='norm', color=[1, 1, 1],
                                          colorSpace='rgb', opacity=1.0,
                                          contrast=1.0, name='myTextStim',
                                          ori=0, depth=0, flipHoriz=False,
                                          flipVert=False, alignHoriz='center',
                                          alignVert='center', bold=False,
                                          italic=False, font='Arial',
                                          fontFiles=[], antialias=True)
        self._screen_objects.append(self._text_stim)

        # other basic components
        self._mouse_handler = event.Mouse(visible=False, win=self._win)

        # finish initialization
        psylog.info('Expyfun: Initialization complete')
        psylog.info('Expyfun: Subject: {0}'
                    ''.format(self._exp_info['participant']))
        psylog.info('Expyfun: Session: {0}'
                    ''.format(self._exp_info['session']))
        self.flush_logs()

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
    def clear_screen(self):
        """Remove all visual stimuli from the screen.
        """
        for comp in self._screen_objects:
            if hasattr(comp, 'setAutoDraw'):
                comp.setAutoDraw(False)
        self._win.callOnFlip(self.write_data_line, 'screen cleared')
        self._win.flip()

    def screen_text(self, text):
        """Show some text on the screen.

        Parameters
        ----------
        text : str
            The text to be rendered.
        """
        self._text_stim.setText(text)
        self._text_stim.setAutoDraw(True)
        self._win.callOnFlip(self.write_data_line, 'screen text', text)
        self._win.flip()

    def screen_prompt(self, text, max_wait=np.inf, min_wait=0, live_keys=None,
                      timestamp=False, clear_screen=True):
        """Display text and (optionally) wait for user continuation

        Parameters
        ----------
        text : str
            The text to display. It will automatically wrap lines.
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
        clear_screen : bool
            If True, ``clear_screen()`` will be called before returning to
            the prompt.

        Returns
        -------
        pressed : tuple | str | None
            If ``timestamp==True``, returns a tuple ``(str, float)`` indicating
            the first key pressed and its timestamp (or ``(None, None)`` if no
            acceptable key was pressed between ``min_wait`` and ``max_wait``).
            If ``timestamp==False``, returns a string indicating the first key
            pressed (or ``None`` if no acceptable key was pressed).
        """
        self.screen_text(text)
        out = self.wait_one_press(max_wait, min_wait, live_keys,
                                  timestamp)
        if clear_screen is True:
            self.clear_screen()
        return out

    def flip_and_play(self):
        """Flip screen, play audio, then run any "on-flip" functions.

        Notes
        -----
        Order of operations is: screen flip, audio start, additional functions
        added with ``on_every_flip``, followed by functions added with
        ``on_next_flip``.
        """
        psylog.info('Expyfun: Flipping screen and playing audio')
        self._win.callOnFlip(self._play)
        if self._on_every_flip is not None:
            for function in self._on_every_flip:
                self._win.callOnFlip(function)
        if self._on_next_flip is not None:
            for function in self._on_next_flip:
                self._win.callOnFlip(function)
        self._win.flip()

    def flip(self):
        """Flip screen, then run any "on-flip" functions.

        Notes
        -----
        Order of operations is: screen flip, audio start, additional functions
        added with ``on_every_flip``, followed by functions added with
        ``on_next_flip``.
        """
        psylog.info('Expyfun: Flipping screen')
        if self._on_every_flip is not None:
            for function in self._on_every_flip:
                self._win.callOnFlip(function)
        if self._on_next_flip is not None:
            for function in self._on_next_flip:
                self._win.callOnFlip(function)
        self._win.flip()

    def call_on_next_flip(self, function, *args, **kwargs):
        """Add a function to be executed on next flip only.

        Notes
        -----
        See ``flip_and_play`` for order of operations. Can be called multiple
        times to add multiple functions to the queue. If function is ``None``,
        will clear all the "on every flip" functions.
        """
        if function is not None:
            function = partial(function, *args, **kwargs)
            if self._on_next_flip is None:
                self._on_next_flip = [function]
            else:
                self._on_next_flip.append(function)
        else:
            self._on_next_flip = None

    def call_on_every_flip(self, function, *args, **kwargs):
        """Add a function to be executed on every flip.

        Notes
        -----
        See ``flip_and_play`` for order of operations. Can be called multiple
        times to add multiple functions to the queue. If function is ``None``,
        will clear all the "on every flip" functions.
        """
        if function is not None:
            function = partial(function, *args, **kwargs)
            if self._on_every_flip is None:
                self._on_every_flip = [function]
            else:
                self._on_every_flip.append(function)
        else:
            self._on_every_flip = None

    def pix2deg(self, xy):
        """Convert pixels to degrees

        Parameters
        ----------
        xy : array-like
            Distances (in pixels) from center to convert to degrees.
        """
        xy = np.asarray(xy)
        return misc.pix2deg(xy, self._win.monitor)

    def deg2pix(self, xy):
        """Convert degrees to pixels

        Parameters
        ----------
        xy : array-like
            Distances (in degrees) from center to convert to pixels.
        """
        xy = np.asarray(xy)
        return misc.deg2pix(xy, self._win.monitor)

    @property
    def on_next_flip_functions(self):
        return self._on_next_flip

    @property
    def on_every_flip_functions(self):
        return self._on_every_flip

    @property
    def window(self):
        return self._win

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

############################# MOUSE METHODS ##################################
    def get_mouse_position(self, units='pix'):
        """Mouse position in screen coordinates

        Parameters
        ----------
        units : str
            Either ``'pix'`` or ``'norm'`` for the type of units to return.

        Returns
        -------
        position : ndarray
            The mouse position.
        """
        if not units in ['pix', 'norm']:
            raise RuntimeError('must request units in "pix" or "norm"')
        pos = np.array(self._mouse_handler.getPos())
        if units == 'pix':
            pos *= self.window.size / 2.
        return pos

    def toggle_cursor(self, visibility, flip=False):
        """Show or hide the mouse

        Parameters
        ----------
        visibility : bool
            If True, show; if False, hide.
        """
        self._win.setMouseVisible(visibility)
        if flip is True:
            self._win.flip()

################################ AUDIO METHODS ###############################
    def start_noise(self):
        """Start the background masker noise."""
        self._ac.start_noise()

    def stop_noise(self):
        """Stop the background masker noise."""
        self._ac.stop_noise()

    def clear_buffer(self):
        """Clear audio data from the audio buffer."""
        self._ac.clear_buffer()
        psylog.info('Expyfun: Buffer cleared')

    def load_buffer(self, samples):
        """Load audio data into the audio buffer.

        Parameters
        ----------
        samples : np.array
            Audio data as floats scaled to (-1,+1), formatted as an Nx1 or Nx2
            numpy array with dtype float32.
        """
        samples = self._validate_audio(samples) * self._stim_scaler
        psylog.info('Expyfun: Loading {} samples to buffer'
                    ''.format(samples.size))
        self._ac.load_buffer(samples)

    def _play(self):
        """Play the audio buffer.
        """
        psylog.debug('Expyfun: playing audio')
        self._ac.play()

    def stop(self):
        """Stop audio buffer playback and reset cursor to beginning of buffer.
        """
        self._ac.stop()
        psylog.info('Expyfun: Audio stopped and reset.')

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
        if self._fs_mismatch is True:
            psylog.warn('Resampling {} seconds of audio'
                        ''.format(round(len(samples) / self._stim_fs), 2))
            num_samples = len(samples) * self.fs / float(self._stim_fs)
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
                warn_string = ('Stimulus max RMS exceeds stated RMS by more '
                               'than 6 dB.')
                psylog.warn(warn_string)
                raise UserWarning(warn_string)
            elif max_rms < 0.5 * self._stim_rms:
                warn_string = ('Stimulus max RMS is less than stated RMS by '
                               'more than 6 dB.')
                psylog.warn(warn_string)
                # raise UserWarning(warn_string)

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
    def write_data_line(self, event, value=None, timestamp=None):
        """Add a line of data to the output CSV.

        Parameters
        ----------
        event : str
            Type of event (e.g., keypress, screen flip, etc.)
        value : None | str
            Anything that can be cast to a string is okay here.
        timestamp : float | None
            The timestamp when the event occurred.  If ``None``, will use the
            time the data line was written from the master clock.

        Notes
        -----
        Writing a data line causes the file to be flushed, which may take
        some time (although it usually shouldn't), so avoid calling during
        critical timing periods.
        """
        if timestamp is None:
            timestamp = self._master_clock.getTime()
        ll = '\t'.join(_sanitize(x) for x in [timestamp, event, value]) + '\n'
        if self._data_file is not None:
            self._data_file.write(ll)
            self._data_file.flush()  # make sure it's actually written out

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
        time_left = timestamp - self._master_clock.getTime()
        if time_left < 0:
            psylog.warn('wait_until was called with a timestamp ({}) that had '
                        'already passed {} seconds prior.'
                        ''.format(timestamp, -time_left))
        else:
            wait_secs(time_left)
        return time_left

    def stamp_triggers(self, trigger_list, delay=0.03):
        """Stamp experiment ID triggers

        Parameters
        ----------
        trigger_list : list
            List of numbers to stamp.
        delay : float
            Delay to use between sequential triggers.

        Notes
        -----
        Depending on how EC was initialized, stamping could be done
        using different pieces of hardware (e.g., parallel port or TDT).
        Also note that it is critical that the input is a list, and
        that all elements are integers. No input checking is done to
        ensure responsiveness.

        Also note that control will not be returned to the script until
        the stamping is complete.
        """
        self._trigger_handler.stamp_triggers(trigger_list, delay)
        psylog.exp('Expyfun: Stamped: ' + str(trigger_list))

    def flush_logs(self):
        """Flush logs (useful for debugging)
        """
        # pyflakes won't like this, but it's better here than outside class
        psylog.flush()

    def close(self):
        """Close all connections in experiment controller.
        """
        self.__exit__(None, None, None)

    def __enter__(self):
        psylog.debug('Expyfun: Entering')
        return self

    def __exit__(self, err_type, value, traceback):
        """
        Notes
        -----
        err_type, value and traceback will be None when called by self.close()
        """
        psylog.debug('Expyfun: Exiting cleanly')

        cleanup_actions = [self.stop_noise, self.stop,
                           self._ac.halt, self._win.close]
        if self._data_file is not None:
            cleanup_actions.append(self._data_file.close)
        cleanup_actions.extend(self._extra_cleanup_fun)
        for action in cleanup_actions:
            try:
                action()
            except Exception as exc:
                print exc
                continue
        try:
            core.quit()
        except SystemExit:
            pass
        except Exception as exc:
            print exc
        if any([x is not None for x in (err_type, value, traceback)]):
            raise err_type, value, traceback

############################# READ-ONLY PROPERTIES ###########################
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
        return self._master_clock.getTime()

    @property
    def _fs_mismatch(self):
        """Quantify if sample rates substantively differ.
        """
        return not np.allclose(self.stim_fs, self.fs, rtol=0, atol=0.05)


def _get_dev_db(audio_controller):
    """Selects device-specific amplitude to ensure equivalence across devices.
    """
    if audio_controller == 'RM1':
        return 108  # this is approx w/ knob @ 12 o'clock (knob not detented)
    elif audio_controller == 'RP2':
        return 108
    elif audio_controller == 'RZ6':
        return 114
    elif audio_controller == 'psychopy':
        return 90  # TODO: this value not yet calibrated, may vary by system
    else:
        psylog.warn('Unknown audio controller: stim scaler may not work '
                    'correctly. You may want to remove your headphones if this'
                    ' is the first run of your experiment.')
        return 90  # for untested TDT models
