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
from psychopy import visual, core, data, event, sound, gui, parallel  # prefs
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
        Keyboard key(s) to utilize as an experiment force-quit button.
        Can be a zero-element list for no force quit support. If None, defaults
        to ['escape', 'lctrl', 'rctrl']
    participant : str | None
        If None, a GUI will be used to acquire this information.
    session : str | None
        If None, a GUI will be used to acquire this information.
    trigger_controller : str | None
        If None, the type will be read from the system configuration file.
        If a string, must be 'dummy', 'parallel', or 'tdt'. Note that by
        default the mode is 'dummy', since setting up the parallel port
        can be a pain.
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

        if force_quit is None:
            force_quit = ['escape', 'lctrl', 'rctrl']

        self._stim_fs = stim_fs
        self._stim_rms = stim_rms
        self._stim_db = stim_db
        self._noise_db = noise_db
        self._force_quit = force_quit

        # Check Pyglet version for safety
        _check_pyglet_version(raise_error=True)

        # some hardcoded parameters...
        bkgd_color = [-1, -1, -1]  # psychopy does RGB from -1 to 1
        root_dir = os.getcwd()

        # dictionary for experiment metadata
        self._exp_info = {'participant': participant, 'session': session,
                          'exp_name': exp_name, 'date': data.getDateStr()}

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
            if session_dialog.OK is False:
                self.close()  # user pressed cancel

        # initialize log file
        if not op.isdir(op.join(root_dir, output_dir)):
            os.mkdir(op.join(root_dir, output_dir))
        basename = op.join(root_dir, output_dir, '%s_%s'
                           % (self._exp_info['participant'],
                              self._exp_info['date']))
        self._log_file = basename + '.log'
        psylog.LogFile(self._log_file, level=psylog.INFO)

        # clocks
        self.master_clock = core.Clock()
        self.trial_clock = core.Clock()

        # list of entities to draw / clear from the visual window
        self._screen_objects = []

        # response device
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
            _noise = np.random.normal(0, 0.01, self._fs * 5.0)  # 5 secs
            self._noise_array = np.array((_noise, -1.0 * _noise), order='F').T
            self._noise = sound.Sound(self._noise_array, sampleRate=self._fs)
            self._noise.setVolume(1.0)  # do not change: don't know if linear
        else:
            raise ValueError('audio_controller[\'TYPE\'] must be '
                             '\'psychopy\' or \'tdt\'.')

        # audio scaling factor; ensure uniform intensity across output devices
        self.set_stim_db(self._stim_db)
        self.set_noise_db(self._noise_db)

        if stim_fs != self._fs:
            psylog.warn('Mismatch between reported stim sample rate ({0}) and '
                        'device sample rate ({1}). ExperimentController will '
                        'resample for you, but that takes a non-trivial amount'
                        ' of processing time and may compromise your '
                        'experimental timing and/or introduce artifacts.'
                        ''.format(stim_fs, self._fs))

        if trigger_controller is None:
            trigger_controller = get_config('TRIGGER_CONTROLLER', 'dummy')
        if trigger_controller == 'tdt':
            self._trigger_handler = self._tdt
        elif trigger_controller == 'parallel':
            self._trigger_handler = _psych_parallel()
        elif trigger_controller == 'dummy':
            self._trigger_handler = _psych_parallel(dummy_mode=True)
        else:
            raise ValueError('trigger_controller must be either "parallel", '
                             '"dummy", or "tdt", not '
                             '{0}'.format(trigger_controller))

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

        # other basic components
        self._button_handler = event.BuilderKeyResponse()
        self._data_handler = data.ExperimentHandler(name=exp_name, version='',
                                                    extraInfo=self._exp_info,
                                                    runtimeInfo=None,
                                                    originPath=None,
                                                    savePickle=True,
                                                    saveWideText=True,
                                                    dataFileName=basename)

        psylog.info('Expyfun: Initialization complete')
        psylog.info('Expyfun: Subject: {0}'
                    ''.format(self._exp_info['participant']))
        psylog.info('Expyfun: Session: {0}'
                    ''.format(self._exp_info['session']))
        self.flush_logs()
        self.master_clock.reset()

    def __repr__(self):
        """Return a useful string representation of the experiment
        """
        string = ('<ExperimentController ({3}): "{0}" {1} ({2})>'
                  ''.format(self._exp_info['exp_name'],
                            self._exp_info['participant'],
                            self._exp_info['session'],
                            self._audio_type))
        return string

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
            If an empty list, all buttons / keys will be accepted.  If None,
            the prompt displays until max_wait seconds have passed.

        Returns
        -------
        pressed : tuple
            A tuple (str, float) indicating the first key pressed and its
            timestamp. If no acceptable key is pressed between min_wait and
            max_wait, returns ([], max_wait).
        """
        if np.isinf(max_wait) and live_keys is None:
            raise ValueError('You have asked for max_wait=inf with '
                             'live_keys=None, this will stall the experiment '
                             'forever.')
        self.screen_text(text)
        if live_keys is None:
            wait_secs(max_wait)
            return (None, max_wait)
        else:
            return self.get_first_press(max_wait, min_wait, live_keys,
                                        timestamp)

    def screen_text(self, text, clock=None):
        """Show some text on the screen.

        Parameters
        ----------
        text : str
            The text to be rendered
        clock : Instance of psychopy.core.Clock()
            Defaults to using self.trial_clock, but could be specified as
            self.master_clock or any other PsychoPy clock object.
        """
        if clock is None:
            clock = self.trial_clock
        self._text_stim.setText(text)
        self._text_stim.tStart = clock.getTime()
        self._text_stim.setAutoDraw(True)
        self._win.flip()

    def clear_screen(self):
        """Remove all visual stimuli from the screen.
        """
        for comp in self._screen_objects:
            if hasattr(comp, 'setAutoDraw'):
                comp.setAutoDraw(False)
        self._win.flip()

    def init_trial(self):
        """Reset trial clock, clear stored keypresses and reaction times.
        """
        self.flush_logs()
        self._button_handler.keys = []
        self._button_handler.rt = []
        self.trial_clock.reset()

    def add_data_line(self, data_dict):
        """Add a line of data to the output CSV.

        Parameters
        ----------
        data_dict : dict
            Key(s) of data_dict determine the column heading of the CSV under
            which the value(s) will be written.
        """
        for key, value in data_dict.items():
            self._data_handler.addData(key, value)
        self._data_handler.nextEntry()

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

    def get_key_buffer(self):
        """Get the entire keyboard / button box buffer.
        """
        if self._response_device == 'keyboard':
            return event.getKeys(timeStamped=False)
        else:
            return self._tdt.get_key_buffer()

    def get_first_press(self, max_wait=np.inf, min_wait=0.0, live_keys=None,
                        timestamp=True):
        """Returns only the first button pressed after min_wait.

        Parameters
        ----------
        max_wait : float
            Duration after which control is returned if no key is pressed.
        min_wait : float
            Duration for which to ignore keypresses.
        live_keys : list | None
            List of strings indicating acceptable keys or buttons. Other data
            types are automatically cast as strings.
        timestamp : bool
            Whether the keypresses should be timestamped.

        Returns
        -------
        pressed : tuple | str | None
            If timestamp==True, returns a tuple (str, float) indicating the
            first key pressed and its timestamp (or (None, None) if no
            acceptable key was pressed between min_wait and max_wait). If
            timestamp==False, returns a string indicating the first key pressed
            (or None if no acceptable key was pressed).
        """
        assert min_wait < max_wait
        if self._response_device == 'keyboard':
            self._button_handler.keys = []
            self._button_handler.rt = []
            self._button_handler.clock.reset()
            wait_secs(min_wait)
            self._check_force_quit()
            event.clearEvents('keyboard')
            live_keys = _add_escape_keys(live_keys, self._force_quit)
            pressed = []
            while (not len(pressed) and
                   self._button_handler.clock.getTime() < max_wait):
                if timestamp:
                    pressed = event.getKeys(keyList=live_keys,
                                        timeStamped=self._button_handler.clock)
                else:
                    pressed = event.getKeys(keyList=live_keys,
                                            timeStamped=False)
            if not len(pressed):
                if timestamp:
                    pressed = (None, None)
                else:
                    pressed = None
            else:
                pressed = pressed[0]  # only keep first press if multiple
            if timestamp:
                self._check_force_quit(pressed[0])
                self._button_handler.keys = pressed[0]
                self._button_handler.rt = pressed[1]
                self._data_handler.addData('button_presses',
                                           self._button_handler.keys)
                self._data_handler.addData('reaction_times',
                                           self._button_handler.rt)
            else:
                self._check_force_quit(pressed)
                self._button_handler.keys = pressed
                self._data_handler.addData('button_presses',
                                           self._button_handler.keys)
            self._data_handler.nextEntry()
            return pressed
        else:
            # TODO: make keyboard escape keys active here
            return self._tdt.get_first_press(max_wait, min_wait, live_keys,
                                             timestamp)

    def get_presses(self, max_wait, min_wait=0.0, live_keys=None,
                    timestamp=True):
        """Returns all button presses between min_wait and max_wait.

        Parameters
        ----------
        max_wait : float
            Duration after which control is returned.
        min_wait : float
            Duration for which to ignore keypresses.
        live_keys : list | None
            List of strings indicating acceptable keys or buttons. Other data
            types are automatically cast as strings.
        timestamp : bool
            Whether the keypresses should be timestamped.

        Returns
        -------
        presses : list
            If timestamp==False, returns a list of strings indicating which
            keys were pressed. Otherwise, returns a list of tuples
            (str, float) of keys and their timestamps. If no keys are pressed,
            returns [].
        """
        assert min_wait < max_wait
        if self._response_device == 'keyboard':
            self._button_handler.keys = []
            self._button_handler.rt = []
            self._button_handler.clock.reset()
            wait_secs(min_wait)
            self._check_force_quit()
            event.clearEvents('keyboard')
            live_keys = _add_escape_keys(live_keys, self._force_quit)
            pressed = []
            while self._button_handler.clock.getTime() < max_wait:
                if timestamp:
                    pressed += event.getKeys(keyList=live_keys,
                                     timeStamped=self._button_handler.clock)
                else:
                    pressed += event.getKeys(keyList=live_keys,
                                             timeStamped=False)
            if len(pressed):
                if timestamp:
                    self._check_force_quit([key for (key, _) in pressed])
                else:
                    self._check_force_quit(pressed)
            for p in pressed:
                if timestamp:
                    self._button_handler.keys = p[0]
                    self._button_handler.rt = p[1]
                    self._data_handler.addData('reaction_times',
                                               self._button_handler.rt)
                else:
                    self._button_handler.keys = p
                self._data_handler.addData('button_presses',
                                           self._button_handler.keys)
                #self._data_handler.addData('trial', trial_num)
                self._data_handler.nextEntry()
            return pressed
        else:
            # TODO: make keyboard escape keys active here
            self._tdt.get_presses(max_wait, min_wait, live_keys, timestamp)

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

    def clear_buffer(self):
        """Clear audio data from the audio buffer.
        """
        if self._tdt is not None:
            self._tdt.clear_buffer()
        else:
            self._audio.setSound(np.zeros((1, 2)))
        psylog.info('Expyfun: Buffer cleared')

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

    def close(self):
        """Close all connections in experiment controller.
        """
        self.__exit__(None, None, None)

    def flip_and_play(self, clock=None):
        """Flip screen and immediately begin playing audio.

        Parameters
        ----------
        clock : Instance of psychopy.core.Clock()
            Defaults to using self.trial_clock, but could be specified as
            self.master_clock or any other PsychoPy clock object.
        """
        psylog.info('Expyfun: Flipping screen and playing audio')
        if clock is None:
            clock = self.trial_clock
        self._win.callOnFlip(self._play, clock)
        if self._fp_function is not None:
            # PP automatically appends calls
            self._win.callOnFlip(self._fp_function)
        self._win.flip()

    def call_on_flip_and_play(self, function, *args, **kwargs):
        """Locus for additional functions to be executed on flip and play.
        """
        if function is not None:
            self._fp_function = partial(function, *args, **kwargs)
        else:
            self._fp_function = None

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

    def set_stim_db(self, new_db):
        """Set the level of the stimuli.
        """
        self._stim_db = new_db
        self._stim_scaler = self._update_sound_scaler(new_db, self._stim_rms)
        # not immediate: new value is applied on the next load_buffer call

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
        psylog.exp('Stamped: ' + str(trigger_list))

    def flush_logs(self):
        """Flush logs (useful for debugging)
        """
        # pyflakes won't like this, but it's better here than outside class
        psylog.flush()

    def _play(self, clock):
        """Play the audio buffer.
        """
        psylog.debug('Expyfun: playing audio')
        if self._tdt is not None:
            self._tdt.play()
        else:
            self._audio.tStart = clock.getTime()
            self._audio.play()
            self.stamp_triggers([1])

    def _update_sound_scaler(self, desired_db, orig_rms):
        """Calcs coefficient ensuring stim ampl equivalence across devices.
        """
        if self._audio_type == 'tdt':
            ac = self._tdt.model
        else:
            ac = self._audio_type
        exponent = (-(_get_dev_db(ac) - desired_db) / 20.0)
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
            The audio samples.
        """
        # check data type
        if type(samples) is list:
            samples = np.asarray(samples, dtype='float32', order='C')
        elif samples.dtype != 'float32':
            samples = np.float32(samples)

        # check values
        if np.max(np.abs(samples)) > 1:
            raise ValueError('Sound data exceeds +/- 1.')
            # samples /= np.max(np.abs(samples),axis=0)

        # check dimensionality
        if samples.ndim == 1:
            samples = np.array((samples, samples)).T
        elif samples.ndim > 2:
            raise ValueError('Sound data has more than two dimensions.')
        elif 1 in samples.shape:
            samples = samples.ravel()
            samples = np.array((samples, samples)).T
        elif 2 not in samples.shape:
            raise ValueError('Sound data has more than two channels.')
        elif samples.shape[0] == 2:
            samples = samples.T

        # check sample rates
        if self._stim_fs != self._fs:
            num_samples = len(samples) * self._fs / float(self._stim_fs)
            samples = resample(samples, int(num_samples), window='boxcar')
        return samples

    def _halt(self):
        """Cleanup action for halting the running circuit on a TDT.
        """
        if self._tdt is not None:
            self._tdt.halt_circuit()

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

        cleanup_actions = (self._win.close, self.stop_noise, self.stop,
                           self._halt)
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


class _psych_parallel(object):
    """Parallel port and dummy triggering support

    Parameters
    ----------
    dummy_mode : bool
        If True, then just pass through stamping calls.
    high_duration : float
        Amount of time (seconds) to leave the trigger high whenever
        sending a trigger.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see expyfun.verbose).

    Notes
    -----
    On Linux, this may require some combination of the following:

    1. ``sudo modprobe ppdev``
    2. ``sudo mknod /dev/parport0 c 99 0 -m 666``
    3. Add user to ``lp`` group (``/etc/group``)
    """
    @verbose_dec
    def __init__(self, dummy_mode=True, high_duration=0.01, verbose=None):
        if dummy_mode is True:
            self.parallel = None
            self.stamp_triggers = self._dummy_triggers
            psylog.info('Initializing dummy triggering mode')
        else:
            self.stamp_triggers = self._stamp_triggers
            # Psychopy has some legacy methods (e.g., parallel.setData()),
            # but we triage here to save time when time-critical stamping
            # may be used
            if platform.system() == 'Linux':
                try:
                    import parallel as _p
                    assert _p
                except ImportError:
                    raise ImportError('must have module "parallel" installed '
                                      'to use parallel triggering on Linux')
                else:
                    self.parallel = parallel.PParallelLinux()
            else:
                raise NotImplementedError

            psylog.info('Initializing psychopy parallel port triggering')
        self.high_duration = high_duration

    def _dummy_triggers(self, triggers, delay):
        pass

    def _stamp_triggers(self, triggers, delay):
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
            self.parallel.setData(int(trig))
            wait_secs(self.high_duration)
            self.parallel.setData(0)
            if ti < len(triggers):
                wait_secs(delay - self.high_duration)


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


def _add_escape_keys(live_keys, _force_quit):
    """Helper to add force quit keys to button press listener.
    """
    if live_keys is not None:
        live_keys = [str(x) for x in live_keys]  # accept ints
        if _force_quit is not None:
            if len(_force_quit) and len(live_keys):
                live_keys = live_keys + _force_quit
    return live_keys
