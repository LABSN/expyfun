"""Tools for controlling experiment execution"""

# Authors: Dan McCloy <drmccloy@uw.edu>
#          Eric Larson <larsoner@uw.edu>
#
# License: BSD (3-clause)

import platform
import numpy as np
import os
from os import path as op
from functools import partial
from scipy.signal import resample
from psychopy import visual, core, event, sound, gui, parallel  # prefs
from psychopy.data import getDateStr as date_str
from psychopy import logging as psylog
from psychopy.constants import STARTED, STOPPED
from .utils import get_config, verbose_dec, _check_pyglet_version, wait_secs
from .tdt_controller import TDTController

# prefs.general['audioLib'] = ['pyo']
# TODO: contact PsychoPy devs to get choice of audio backend
# TODO: PsychoPy expose pygame "loops" argument


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
        Can be 'keyboard' or 'buttonbox'.  If None, the type will be read
        from the machine configuration file.
    stim_rms : float
        The RMS amplitude that the stimuli were generated at (strongly
        recommended to be 0.01).
    stim_db : float
        The desired dB SPL at which to play the stimuli.
    noise_db : float
        The desired dB SPL at which to play the dichotic noise.
    output_dir : str | 'rawData'
        An absolute or relative path to a directory in which raw experiment
        data will be stored. If output_folder does not exist, it will be
        created.
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
                 trigger_controller=None, session=None, verbose=None):

        # Check Pyglet version for safety
        _check_pyglet_version(raise_error=True)

        # assure proper formatting for force-quit keys
        if force_quit is None:
            force_quit = ['lctrl', 'rctrl']
        elif isinstance(force_quit, (int, basestring)):
            force_quit = [str(force_quit)]
        if 'escape' in force_quit:
            psylog.warn('Expyfun: using "escape" as a force-quit key is not '
                        'recommended because it has special status in pyglet.')
        self._force_quit = force_quit

        # initialize some values
        self.trial_id = None
        self._stim_fs = stim_fs
        self._stim_rms = stim_rms
        self._stim_db = stim_db
        self._noise_db = noise_db
        self._stim_scaler = None
        # clocks
        self._master_clock = core.MonotonicClock()
        # list of entities to draw / clear from the visual window
        self._screen_objects = []
        # placeholder for extra actions to do on flip-and-play
        self._on_every_flip = None
        self._on_next_flip = None
        # some hardcoded parameters...
        bkgd_color = [-1, -1, -1]  # psychopy does RGB from -1 to 1
        root_dir = os.getcwd()

        # dictionary for experiment metadata
        self._exp_info = {'participant': participant, 'session': session,
                          'exp_name': exp_name, 'date': date_str()}

        # session start dialog, if necessary
        fixed_list = ['exp_name', 'date']  # things not user-editable in GUI
        for key, value in self._exp_info.iteritems():
            if key not in fixed_list and value is not None:
                if not isinstance(value, basestring):
                    raise TypeError('{} must be string or None'.format(value))
                fixed_list += [key]

        if len(fixed_list) < len(self._exp_info):
            session_dialog = gui.DlgFromDict(dictionary=self._exp_info,
                                             fixed=fixed_list,
                                             title=exp_name)
            if not session_dialog.OK:
                self.close()  # user pressed cancel

        # initialize log file
        if not op.isdir(op.join(root_dir, output_dir)):
            os.mkdir(op.join(root_dir, output_dir))
        basename = op.join(root_dir, output_dir, '%s_%s'
                           % (self._exp_info['participant'],
                              self._exp_info['date']))
        self._log_file = basename + '.log'
        psylog.LogFile(self._log_file, level=psylog.INFO)

        # initialize data file
        self._data_file = open(basename + '.tab', 'a')
        self._data_file.write('# ' + str(self._exp_info) + '\n')
        self._data_file.write('timestamp\tevent\tvalue\n')

        # set up response device
        if response_device is None:
            self._response_device = get_config('RESPONSE_DEVICE', 'keyboard')
        else:
            self._response_device = response_device

        # set audio type
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

        # initialize audio
        if self._audio_type == 'tdt':
            psylog.info('Expyfun: Setting up TDT')
            self._tdt = TDTController(audio_controller)
            self._audio_type = self._tdt.model
            self._fs = self._tdt.fs
        elif self._audio_type == 'psychopy':
            psylog.info('Expyfun: Setting up PsychoPy audio with {} '
                        'backend'.format(sound.audioLib))
            self._tdt = None
            self._fs = 44100
            if sound.Sound is None:
                raise ImportError('PsychoPy sound could not be initialized. '
                                  'Ensure you have the pygame package properly'
                                  ' installed.')
            self._audio = sound.Sound(np.zeros((1, 2)), sampleRate=self._fs)
            self._audio.setVolume(1.0)  # do not change: don't know if linear
            _noise = np.random.normal(0, 0.01, self._fs * 15.0)  # 15 secs
            self._noise_array = np.array(np.c_[_noise, -1.0 * _noise],
                                         order='C')
            self._noise = sound.Sound(self._noise_array, sampleRate=self._fs)
            self._noise.setVolume(1.0)  # do not change: don't know if linear
        else:
            raise ValueError('audio_controller[\'TYPE\'] must be '
                             '\'psychopy\' or \'tdt\'.')
        if stim_fs != self._fs:
            psylog.warn('Mismatch between reported stim sample rate ({0}) and '
                        'device sample rate ({1}). ExperimentController will '
                        'resample for you, but that takes a non-trivial amount'
                        ' of processing time and may compromise your '
                        'experimental timing and/or introduce artifacts.'
                        ''.format(stim_fs, self._fs))

        # audio scaling factor; ensure uniform intensity across output devices
        self.set_stim_db(self._stim_db)
        self.set_noise_db(self._noise_db)

        # set up trigger controller
        if trigger_controller is None:
            trigger_controller = get_config('TRIGGER_CONTROLLER', 'dummy')
        if isinstance(trigger_controller, basestring):
            trigger_controller = dict(type=trigger_controller)
        if trigger_controller['type'] == 'tdt':
            self._trigger_handler = self.tdt
        elif trigger_controller['type'] in ['parallel', 'dummy']:
            if 'address' not in trigger_controller['type']:
                trigger_controller['address'] = get_config('TRIGGER_ADDRESS')
            out = _PsychTrigger(trigger_controller['type'],
                                trigger_controller.get('address'))
            self._trigger_handler = out
        else:
            raise ValueError('trigger_controller type must be '
                             '"parallel", "dummy", or "tdt", not '
                             '{0}'.format(trigger_controller['type']))
        self._trigger_controller = trigger_controller['type']

        # placeholder for extra actions to do on flip-and-play
        self._fp_function = None

        # create visual window
        psylog.info('Expyfun: Setting up screen')
        if window_size is None:
            window_size = get_config('WINDOW_SIZE', '1920,1080').split(',')
        if screen_num is None:
            screen_num = int(get_config('SCREEN_NUM', '0'))
        self._win = visual.Window(size=window_size, fullscr=full_screen,
                                  monitor='', screen=screen_num,
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
        #self.shape_stim = visual.ShapeStim()
        #self._screen_objects.append(self.shape_stim)

        # set up timing
        self._master_clock = core.MonotonicClock()
        self._listen_time = None
        self._time_correction = None
        self._time_correction = self._get_time_correction()

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
                      timestamp=False):
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

        Returns
        -------
        pressed : tuple | str | None
            If ``timestamp==True``, returns a tuple ``(str, float)`` indicating the
            first key pressed and its timestamp (or ``(None, None)`` if no
            acceptable key was pressed between ``min_wait`` and ``max_wait``). If
            ``timestamp==False``, returns a string indicating the first key pressed
            (or ``None`` if no acceptable key was pressed).
        """
        if np.isinf(max_wait) and live_keys == []:
            raise ValueError('You have asked for max_wait=inf with '
                             'live_keys=[], this will stall the experiment '
                             'forever.')
        self.screen_text(text)
        if live_keys == []:
            wait_secs(max_wait)
            return (None, max_wait)
        else:
            return self.wait_one_press(max_wait, min_wait, live_keys,
                                       timestamp)

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

    @property
    def on_next_flip_functions(self):
        return self._on_next_flip

    @property
    def on_every_flip_functions(self):
        return self._on_every_flip

############################ KEY / BUTTON METHODS ############################
    def listen_presses(self):
        """Start listening for keypresses.
        """
        self._time_correction = self._get_time_correction()
        self._listen_time = self._master_clock.getTime()
        event.clearEvents('keyboard')
        # TODO: clear TDT button box events, add if self._response_device ?

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
        pressed = []
        if timestamp and relative_to is None:
            if self._listen_time is None:
                raise ValueError('I cannot timestamp: relative_to is None and '
                                 'you have not yet called listen_presses.')
            else:
                relative_to = self._listen_time
        if self._response_device == 'keyboard':
            pressed = event.getKeys(live_keys, timeStamped=True)
        else:
            pressed = self._tdt.get_key_buffer()  # TODO: implement

        if len(pressed):
            pressed = [(k, s + self._time_correction) for k, s in pressed]
            self._log_presses(pressed)
            keys = [k for k, _ in pressed]
            self._check_force_quit(keys)
            if timestamp:
                pressed = [(k, s - relative_to) for k, s in pressed]
            else:
                pressed = keys
        return pressed

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
            Whether the keypress should be timestamped. If ``True``, returns the
            button press time relative to the value given in ``relative_to``.
        relative_to : None | float
            A time relative to which timestamping is done. Ignored if
            timestamp==False.  If ``None``, timestamps are relative to the time
            ``wait_one_press`` was called.

        Returns
        -------
        pressed : tuple | str | None
            If timestamp==True, returns a tuple (str, float) indicating the
            first key pressed and its timestamp (or (None, None) if no
            acceptable key was pressed between min_wait and max_wait). If
            timestamp==False, returns a string indicating the first key pressed
            (or None if no acceptable key was pressed).
        """
        relative_to, start_time = self._init_wait_press(max_wait, min_wait,
                                                        live_keys, timestamp,
                                                        relative_to)

        if self._response_device == 'keyboard':
            live_keys = self._add_escape_keys(live_keys)
            pressed = []
            while (not len(pressed) and
                   self._master_clock.getTime() - start_time < max_wait):
                pressed = event.getKeys(keyList=live_keys,
                                        timeStamped=self._master_clock)
        else:
            pressed = self._tdt.wait_one_press(max_wait, min_wait, live_keys,
                                               timestamp, relative_to)
        if len(pressed):
            self._log_presses(pressed)  # multiple presses all get logged
            key = pressed[0][0]  # only first press is returned
            stamp = pressed[0][1]
            self._check_force_quit(key)
            if timestamp:
                pressed = (key, stamp - relative_to)
            else:
                pressed = key
        # handle non-presses
        elif timestamp:
            pressed = (None, None)
        else:
            pressed = None
        return pressed

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
            Whether the keypresses should be timestamped. If ``True``, returns the
            button press time relative to the value given in ``relative_to``.
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
        relative_to, start_time = self._init_wait_press(max_wait, min_wait,
                                                        live_keys, timestamp,
                                                        relative_to)

        if self._response_device == 'keyboard':
            live_keys = self._add_escape_keys(live_keys)
            pressed = []
            while (self._master_clock.getTime() - start_time < max_wait):
                pressed += event.getKeys(keyList=live_keys,
                                         timeStamped=self._master_clock)
        else:
            pressed += self._tdt.wait_one_press(max_wait, min_wait, live_keys,
                                                timestamp, relative_to)
        if len(pressed):
            self._log_presses(pressed)
            keys = [k for k, _ in pressed]
            self._check_force_quit(keys)
            if timestamp:
                pressed = [(k, s - relative_to) for k, s in pressed]
            else:
                pressed = keys
        return pressed

    def _log_presses(self, pressed):
        """Write key presses to data file.
        """
        for key, stamp in pressed:
            self.write_data_line('keypress', key, stamp)

    def _init_wait_press(self, max_wait, min_wait, live_keys, timestamp,
                         relative_to):
        """Actions common to ``wait_one_press`` and ``wait_for_presses``

        Parameters
        ----------
        max_wait : float
            Duration after which control is returned.
        min_wait : float
            Duration for which to ignore keypresses (force-quit keys will
            still be checked at the end of the wait).
        """
        if np.isinf(max_wait) and live_keys == []:
            raise ValueError('max_wait cannot be infinite if there are no live'
                             ' keys.')
        if not min_wait < max_wait:
            raise ValueError('min_wait must be less than max_wait')
        start_time = self._master_clock.getTime()
        if timestamp and relative_to is None:
            relative_to = start_time
        wait_secs(min_wait)
        self._check_force_quit()
        event.clearEvents('keyboard')
        return relative_to, start_time

    def _add_escape_keys(self, live_keys):
        """Helper to add force quit keys to button press listener.
        """
        if live_keys is not None:
            live_keys = [str(x) for x in live_keys]  # accept ints
            if len(self._force_quit):  # should always be a list of strings
                live_keys = live_keys + self._force_quit
        return live_keys

    def _check_force_quit(self, keys=None):
        """Compare key buffer to list of force-quit keys and quit if matched.

        Parameters
        ----------
        keys : str | list | None
            List of keypresses to check against self._force_quit.
        """
        if keys is None:
            keys = event.getKeys(self._force_quit, timeStamped=False)
        elif type(keys) is str:
            keys = [k for k in [keys] if k in self._force_quit]
        elif type(keys) is list:
            keys = [k for k in keys if k in self._force_quit]
        else:
            raise TypeError('Force quit checking requires a string or list of'
                            ' strings, not a {}.'.format(type(keys)))
        if len(keys):
            self.close()

################################ AUDIO METHODS ###############################
    def start_noise(self):
        """Start the background masker noise.
        """
        if self._tdt is not None:
            self._tdt.start_noise()
        else:
            #self._audio.start_noise()
            self._noise._snd.play(loops=-1)
            self._noise.status = STARTED
            psylog.info('Expyfun: Started noise (PsychoPy)')

    def stop_noise(self):
        """Stop the background masker noise.
        """
        if self._tdt is not None:
            self._tdt.stop_noise()
        else:
            #self._audio.stop_noise()
            self._noise.stop()
            self._noise.status = STOPPED
            psylog.info('Expyfun: Stopped noise (PsychoPy)')

    def clear_buffer(self):
        """Clear audio data from the audio buffer.
        """
        if self._tdt is not None:
            self._tdt.clear_buffer()
        else:
            self._audio.setSound(np.zeros((1, 2)))
        psylog.info('Expyfun: Buffer cleared')

    def load_buffer(self, samples):
        """Load audio data into the audio buffer.

        Parameters
        ----------
        samples : np.array
            Audio data as floats scaled to (-1,+1), formatted as an Nx1 or Nx2
            numpy array with dtype float32.
        """
        samples = self._validate_audio(samples)
        psylog.info('Expyfun: Loading {} samples to buffer'
                    ''.format(samples.size))
        if self._tdt is not None:
            self._tdt.load_buffer(samples * self._stim_scaler)
        else:
            self._audio = sound.Sound(samples * self._stim_scaler,
                                      sampleRate=self._fs)
            self._audio.setVolume(1.0)  # do not change: don't know if linear

    def _play(self):
        """Play the audio buffer.
        """
        psylog.debug('Expyfun: playing audio')
        if self._tdt is not None:
            self._tdt.play()
        else:
            self._audio.play()
            self.stamp_triggers([1])

    def stop(self):
        """Stop audio buffer playback and reset cursor to beginning of buffer.
        """
        if self._tdt is not None:
            self._tdt.stop()
            self._tdt.reset()
        else:
            # PsychoPy doesn't cleanly support playing from middle, so no
            # rewind necessary (it happens automatically)
            self._audio.stop()
        psylog.info('Expyfun: Audio stopped and reset.')

    def set_stim_db(self, new_db):
        """Set the level of the stimuli.
        """
        self._stim_db = new_db
        self._stim_scaler = self._update_sound_scaler(new_db, self._stim_rms)
        # not immediate: new value is applied on the next load_buffer call

    def set_noise_db(self, new_db):
        """Set the level of the background noise.
        """
        if self._tdt is not None:
            # Our TDT circuit generates noise at RMS of 1 (as opposed to 0.01
            # for the python-generated noise in __init__.)
            _noise_scaler = self._update_sound_scaler(new_db, 1.0)
            self._tdt.set_noise_level(_noise_scaler)
        else:
            _noise_scaler = self._update_sound_scaler(new_db, 0.01)
            _new_noise = sound.Sound(self._noise_array * _noise_scaler,
                                     sampleRate=self._fs)
            if self._noise.status == STARTED:
                # change the noise level immediately
                self._noise.stop()
                self._noise = _new_noise
                self._noise._snd.play(loops=-1)
                self._noise.status = STARTED  # have to explicitly set status,
                # since we bypass PsychoPy's play() method to access "loops"
            else:
                self._noise = _new_noise
        self._noise_db = new_db

    def _update_sound_scaler(self, desired_db, orig_rms):
        """Calcs coefficient ensuring stim ampl equivalence across devices.
        """
        if self._audio_type == 'tdt':
            ac_type = self._tdt.model
        else:
            ac_type = self._audio_type
        exponent = (-(_get_dev_db(ac_type) - desired_db) / 20.0)
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
        elif len(samples.shape) == 2 and samples.shape[0] == 2:
            samples = samples.T

        # resample if needed
        if self._stim_fs != self._fs:
            psylog.warn('Resampling {} seconds of audio'
                        ''.format(round(len(samples) / self._stim_fs), 2))
            num_samples = len(samples) * self._fs / float(self._stim_fs)
            samples = resample(samples, int(num_samples), window='boxcar')

        # make stereo if not already
        if samples.ndim == 1:
            samples = np.array((samples, samples)).T
        elif 1 in samples.shape:
            samples = samples.ravel()
            samples = np.array((samples, samples)).T

        return np.ascontiguousarray(samples)

    def _halt(self):
        """Cleanup action for halting the running circuit on a TDT.
        """
        if self._tdt is not None:
            self._tdt.halt_circuit()
        else:
            pass

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
        """
        if timestamp is None:
            timestamp = self._master_clock.getTime()
        line = str(timestamp) + '\t' + str(event) + '\t' + str(value) + '\n'
        self._data_file.write(line)

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

    def _get_time_correction(self):
        """Clock correction for pyglet- or TDT-generated timestamps.
        """
        if self._response_device == 'keyboard':
            other_time = 0.0  # TODO: get the pyglet clock
        else:
            other_time = 0.0  # TODO: get TDT clock
            raise NotImplementedError
        start_time = self._master_clock.getTime()
        time_correction = start_time - other_time
        if self._time_correction is not None and \
                np.abs(self._time_correction - time_correction) > 0.001:
            psylog.warn('Expyfun: drift of > 1 ms between pyglet clock and '
                        'experiment master clock.')
        return time_correction

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

        cleanup_actions = [self._data_file.close, self.stop_noise,
                           self.stop, self._halt, self._win.close]
        if self._trigger_controller in ['parallel', 'dummy']:
            cleanup_actions += [self._trigger_handler.close]
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
        return self._fs  # not user-settable

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


class _PsychTrigger(object):
    """Parallel port and dummy triggering support

    Parameters
    ----------
    mode : str
        'parallel' for real use. 'dummy', passes all calls.
    high_duration : float
        Amount of time (seconds) to leave the trigger high whenever
        sending a trigger.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see expyfun.verbose).

    Notes
    -----
    On Linux, parallel port may require some combination of the following:

    1. ``sudo modprobe ppdev``
    2. Add user to ``lp`` group (``/etc/group``)
    3. Run ``sudo rmmod lp`` (otherwise ``lp`` takes exclusive control)
    4. Edit ``/etc/modprobe.d/blacklist.conf`` to add ``blacklist lp``
    """
    @verbose_dec
    def __init__(self, mode='dummy', address=None, high_duration=0.01,
                 verbose=None):
        self.parallel = None
        if mode == 'parallel':
            psylog.info('Initializing psychopy parallel port triggering')
            self._stamp_trigger = self._parallel_trigger
            # Psychopy has some legacy methods (e.g., parallel.setData()),
            # but we triage here to save time when time-critical stamping
            # may be used
            if 'Linux' in platform.system():
                address = '/dev/parport0' if address is None else address
                try:
                    import parallel as _p
                    assert _p
                except ImportError:
                    raise ImportError('must have module "parallel" installed '
                                      'to use parallel triggering on Linux')
                else:
                    self.parallel = parallel.PParallelLinux(address)
            else:
                raise NotImplementedError
        else:  # mode == 'dummy':
            psylog.info('Initializing dummy triggering mode')
            self._stamp_trigger = self._dummy_trigger

        self.high_duration = high_duration

    def _dummy_trigger(self, trig):
        """Fake stamping"""
        wait_secs(self.high_duration)

    def _parallel_trigger(self, trig):
        """Stamp a single byte via parallel port"""
        self.parallel.setData(int(trig))
        wait_secs(self.high_duration)
        self.parallel.setData(0)

    def stamp_triggers(self, triggers, delay):
        """Stamp a list of triggers with a given inter-trigger delay

        Parameters
        ----------
        triggers : list
            No input checking is done, so ensure triggers is a list,
            with each entry an integer with fewer than 8 bits (max 255).
        delay : float
            The inter-trigger delay.
        """
        for ti, trig in enumerate(triggers):
            if ti < len(triggers):
                self._stamp_trigger(trig)
                wait_secs(delay - self.high_duration)

    def close(self):
        """Release hardware interfaces
        """
        if self.parallel is not None:
            del self.parallel


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
