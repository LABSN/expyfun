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
from psychopy import visual, core, data, event, sound, gui, parallel
from psychopy import logging as psylog

from .utils import get_config, verbose_dec, _check_pyglet_version, wait_secs
from .tdt import TDT


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
        (see documentation for expyfun.TDT).
    response_device : str | None
        Can be 'keyboard' or 'buttonbox'.  If None, the type will be read
        from the machine configuration file.
    stim_rms : float
        The RMS amplitude that the stimuli were generated at (strongly
        recommended to be 0.01).
    stim_amp : float
        The desired dB SPL at which to play the stimuli.
    noise_amp : float
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
                 stim_rms=0.01, stim_fs=44100, stim_amp=65, noise_amp=-np.Inf,
                 output_dir='rawData', window_size=None, screen_num=None,
                 full_screen=True, force_quit=None, participant=None,
                 trigger_controller=None, session=None, verbose=None):

        if force_quit is None:
            force_quit = ['escape', 'lctrl', 'rctrl']

        # self._stim_fs = stim_fs
        self._stim_amp = stim_amp
        self._noise_amp = noise_amp
        self._force_quit = force_quit

        # Check Pyglet version for safety
        _check_pyglet_version(raise_error=True)

        # some hardcoded parameters...
        bkgd_color = [-1, -1, -1]  # psychopy does RGB from -1 to 1
        root_dir = os.getcwd()

        # dictionary for experiment metadata
        self.exp_info = {'participant': participant, 'session': session,
                         'exp_name': exp_name, 'date': data.getDateStr()}

        # session start dialog, if necessary
        fixed_list = ['exp_name', 'date']  # things not user-editable in GUI
        for key, value in self.exp_info.iteritems():
            if key not in fixed_list and value is not None:
                if not isinstance(value, basestring):
                    raise TypeError(value + ' must be a string or None')
                fixed_list += [key]

        if len(fixed_list) < len(self.exp_info):
            session_dialog = gui.DlgFromDict(dictionary=self.exp_info,
                                             fixed=fixed_list,
                                             title=exp_name)
            if session_dialog.OK is False:
                self.close()  # user pressed cancel

        # initialize log file
        if not op.isdir(op.join(root_dir, output_dir)):
            os.mkdir(op.join(root_dir, output_dir))
        basename = op.join(root_dir, output_dir, '%s_%s'
                           % (self.exp_info['participant'],
                              self.exp_info['date']))
        self._log_file = basename + '.log'
        psylog.LogFile(self._log_file, level=psylog.INFO)

        # clocks
        self.master_clock = core.Clock()
        self.trial_clock = core.Clock()

        # list of entities to draw / clear from the visual window
        self.screen_objects = []

        # response device
        if response_device is None:
            self.response_device = get_config('RESPONSE_DEVICE', 'keyboard')
        else:
            self.response_device = response_device

        # audio setup
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

        self.audio_type = audio_controller['TYPE'].lower()
        if self.audio_type == 'tdt':
            psylog.info('Expyfun: Setting up TDT')
            self.tdt = TDT(audio_controller)
            self.audio_type = self.tdt.model
            self._fs = self.tdt.fs
        elif self.audio_type == 'psychopy':
            psylog.info('Expyfun: Setting up PsychoPy audio')
            self.tdt = None
            self._fs = 44100
            if sound.Sound is None:
                raise ImportError('PsychoPy sound could not be initialized. '
                                  'Ensure you have the pygame package properly'
                                  ' installed.')
            self.audio = sound.Sound(np.zeros((1, 2)), sampleRate=self._fs)
            self.audio.setVolume(1.0)  # 0 to 1
        else:
            raise ValueError('audio_controller[\'TYPE\'] must be '
                             '\'psychopy\' or \'tdt\'.')

        # scaling factor to ensure uniform intensity across output devices
        self._stim_scaler = _get_stim_scaler(self.audio_type, stim_amp,
                                             stim_rms)

        if trigger_controller is None:
            trigger_controller = get_config('TRIGGER_CONTROLLER', 'dummy')
        if trigger_controller == 'tdt':
            self._trigger_handler = self.tdt
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
        self.win = visual.Window(size=window_size, fullscr=full_screen,
                                 monitor='', screen=screen_num,
                                 winType='pyglet', allowGUI=False,
                                 allowStencil=False, color=bkgd_color,
                                 colorSpace='rgb')
        self.text_stim = visual.TextStim(win=self.win, text='', pos=[0, 0],
                                         height=0.1, wrapWidth=1.2,
                                         units='norm', color=[1, 1, 1],
                                         colorSpace='rgb', opacity=1.0,
                                         contrast=1.0, name='myTextStim',
                                         ori=0, depth=0, flipHoriz=False,
                                         flipVert=False, alignHoriz='center',
                                         alignVert='center', bold=False,
                                         italic=False, font='Arial',
                                         fontFiles=[], antialias=True)
        self.screen_objects.append(self.text_stim)
        #self.shape_stim = visual.ShapeStim()
        #self.screen_objects.append(self.shape_stim)

        # other basic components
        self.button_handler = event.BuilderKeyResponse()
        self.data_handler = data.ExperimentHandler(name=exp_name, version='',
                                                   extraInfo=self.exp_info,
                                                   runtimeInfo=None,
                                                   originPath=None,
                                                   savePickle=True,
                                                   saveWideText=True,
                                                   dataFileName=basename)

        psylog.info('Expyfun: Initialization complete')
        psylog.info('Expyfun: Subject: {0}'
                    ''.format(self.exp_info['participant']))
        psylog.info('Expyfun: Session: {0}'
                    ''.format(self.exp_info['session']))
        self.flush_logs()
        self.master_clock.reset()

    def __repr__(self):
        """Return a useful string representation of the experiment
        """
        string = ('<ExperimentController ({3}): "{0}" {1} ({2})>'
                  ''.format(self.exp_info['exp_name'],
                            self.exp_info['participant'],
                            self.exp_info['session'],
                            self.audio_type))
        return string

    def screen_prompt(self, text, max_wait=np.inf, min_wait=0, live_keys=None):
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
            return self.get_press(max_wait, min_wait, live_keys)

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
        self.text_stim.setText(text)
        self.text_stim.tStart = clock.getTime()
        self.text_stim.setAutoDraw(True)
        self.win.flip()

    def clear_screen(self):
        """Remove all visual stimuli from the screen.
        """
        #self.text_stim.status = FINISHED
        for comp in self.screen_objects:
            if hasattr(comp, 'setAutoDraw'):
                comp.setAutoDraw(False)
        self.win.flip()

    def init_trial(self):
        """Reset trial clock, clear stored keypresses and reaction times.
        """
        self.trial_clock.reset()
        self.button_handler.keys = []
        self.button_handler.rt = []

    def add_data_line(self, data_dict):
        """Add a line of data to the output CSV.

        Parameters
        ----------
        data_dict : dict
            Key(s) of data_dict determine the column heading of the CSV under
            which the value(s) will be written.
        """
        for key, value in data_dict.items():
            self.data_handler.addData(key, value)
        self.data_handler.nextEntry()

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

    def get_press(self, max_wait=np.inf, min_wait=0.0, live_keys=None,
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
        pressed : tuple | str | list
            If timestamp==True, returns a tuple (str, float) indicating the
            first key pressed and its timestamp. If timestamp==False, returns
            a string indicating the first key pressed. If no acceptable key is
            pressed between min_wait and max_wait, returns [].
        """
        assert min_wait < max_wait
        if self.response_device == 'keyboard':
            live_keys = _add_escape_keys(live_keys, self._force_quit)
            self.button_handler.keys = []
            self.button_handler.rt = []
            self.button_handler.clock.reset()
            wait_secs(min_wait)
            event.clearEvents('keyboard')
            pressed = []
            while (not len(pressed) and
                   self.button_handler.clock.getTime() < max_wait):
                if timestamp:
                    pressed = event.getKeys(keyList=live_keys,
                                            timeStamped=self.button_handler.clock)
                else:
                    pressed = event.getKeys(keyList=live_keys,
                                            timeStamped=False)
            if len(pressed):
                pressed = pressed[0]
                self._check_force_quit(pressed)
	        if len(pressed) and timestamp:
	            self.button_handler.keys = pressed[0]
	            self.button_handler.rt = pressed[1]
	            self.data_handler.addData('reaction_times',
	                                      self.button_handler.rt)
	        else:
	            self.button_handler.keys = pressed
	        self.data_handler.addData('button_presses',
	                                  self.button_handler.keys)
	        self.data_handler.nextEntry()
            return pressed
        else:
            raise NotImplementedError()
            # TODO: check the proper tag name for our circuit
            # self.tdt.rpcox.GetTagVal('ButtonBoxTagName')

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
        if self.response_device == 'keyboard':
            live_keys = _add_escape_keys(live_keys, self._force_quit)
            self.button_handler.keys = []
            self.button_handler.rt = []
            self.button_handler.clock.reset()
            pressed = []
            wait_secs(min_wait)
            event.clearEvents('keyboard')
            while self.button_handler.clock.getTime() < max_wait:
                if timestamp:
                    pressed += event.getKeys(keyList=live_keys,
                                             timeStamped=self.button_handler.clock)
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
                    self.button_handler.keys = p[0]
                    self.button_handler.rt = p[1]
                    self.data_handler.addData('reaction_times',
                                              self.button_handler.rt)
                else:
                    self.button_handler.keys = p
                self.data_handler.addData('button_presses',
                                          self.button_handler.keys)
                #self.data_handler.addData('trial', trial_num)
                self.data_handler.nextEntry()
            return pressed
        else:
            raise NotImplementedError()
            # TODO: check the proper tag name for our circuit
            # self.tdt.rpcox.GetTagVal('ButtonBoxTagName')

    def _check_force_quit(self, keys):
        """Compare key buffer to list of force-quit keys and quit if matched.

        Parameters
        ----------
        keys : str | list
            List of keypresses to check against self._force_quit.
        """
        if self._force_quit in keys:
            self.close()

    def load_buffer(self, data, offset=0, buffer_name=None):
        """Load audio data into the audio buffer.

        Parameters
        ----------
        data : np.array
            Audio data as floats scaled to (-1,+1), formatted as an Nx1 or Nx2
            numpy array with dtype float32.
        buffer_name : str | None
            Name of the TDT buffer to target. Ignored if audio_controller is
            'psychopy'.
        """
        psylog.info('Expyfun: Loading {} samples to buffer'.format(data.size))
        if self.tdt is not None:
            self.tdt.write_buffer(buffer_name, offset,
                                  data * self._stim_scaler)
        else:
            if len(data.shape) > 2:
                raise ValueError('Sound data has more than two dimensions.')
            elif len(data.shape) == 2:
                if data.shape[1] > 2:
                    raise ValueError('Sound data has more than two channels.')
                #elif data.shape[1] == 2:
                #    data = data[:, 0]
                elif data.shape[1] == 1:
                    data = data[:, 0]

            self.audio = sound.Sound(np.asarray(data, order='C'),
                                     sampleRate=self._fs)
            self.audio.setVolume(1.0)  # TODO: check this w/r/t stim_scaler

    def clear_buffer(self, buffer_name=None):
        """Clear audio data from the audio buffer.

        Parameters
        ----------
        buffer_name : str | None
            Name of the TDT buffer to target. Ignored if audio_controller is
            'psychopy'.
        """
        psylog.info('Expyfun: Clearing buffer')
        if self.tdt is not None:
            self.tdt.clear_buffer(buffer_name)
        else:
            self.audio.setSound(np.zeros((1, 2)))

    def stop_reset(self):
        """Stop audio buffer playback and reset cursor to beginning of buffer.
        """
        psylog.info('Expyfun: Stopping and resetting audio playback')
        if self.tdt is not None:
            self.tdt.stop()
            self.tdt.reset()
        else:
            # PsychoPy doesn't cleanly support playing from middle, so no
            # rewind necessary
            self.audio.stop()

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
        self.win.callOnFlip(self._play, clock)
        if self._fp_function is not None:
            # PP automatically appends calls
            self.win.callOnFlip(self._fp_function)
        self.win.flip()

    def call_on_flip_and_play(self, function, *args, **kwargs):
        """Locus for additional functions to be executed on flip and play.
        """
        if function is not None:
            self._fp_function = partial(function, *args, **kwargs)
        else:
            self._fp_function = None

    def set_noise_amp(self, new_amp):
        """Set the amplitude of stationary background noise.
        """
        self._noise_amp = new_amp

    def set_stim_amp(self, new_amp):
        """Set the amplitude of the stimuli.
        """
        self._stim_amp = new_amp

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
        if self.tdt is not None:
            self.tdt.play()
        else:
            self.audio.tStart = clock.getTime()
            self.audio.play()
            self.stamp_triggers([1])

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
        self.win.close()
        if self.tdt is not None:
            self.tdt.stop_noise()
            self.stop_reset()
            self.tdt.halt_circuit()
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


def _get_rms(audio_controller):
    """Selects device-specific amplitude to ensure equivalence across devices.
    """
    if audio_controller == 'RM1':
        return 108  # this is approx; knob is not detented
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


def _get_stim_scaler(audio_controller, stim_amp, stim_rms):
    """Calculates coefficient ensuring stim ampl equivalence across devices.
    """
    exponent = (-(_get_rms(audio_controller) - stim_amp) / 20) / stim_rms
    return np.power(10, exponent)


def _add_escape_keys(live_keys, _force_quit):
    """Helper to add force quit keys to button press listener.
    """
    if live_keys is not None:
        live_keys = [str(x) for x in live_keys]  # accept ints
        if _force_quit is not None:
            if len(_force_quit) and len(live_keys):
                live_keys = live_keys + _force_quit
    return live_keys
