"""TODO: add docstring
"""
# import logging
import numpy as np
import os
from os import path as op
from functools import partial
from psychopy import visual, core, data, event, sound, gui
from psychopy import logging as psylog
#from psychopy.constants import FINISHED, STARTED, NOT_STARTED

from .utils import get_config, verbose
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
        Can be a zero-element list for no force quit support.
    participant : str | None
        If None, a GUI will be used to acquire this information.
    session : str | None
        If None, a GUI will be used to acquire this information.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see expyfun.verbose).

    Returns
    -------
    exp_controller : instance of ExperimentController
        The experiment control interface.

    Notes
    -----
    TODO: add eye tracker and EEG
    """

    @verbose
    def __init__(self, exp_name, audio_controller=None, response_device=None,
                 stim_rms=0.01, stim_fs=44100, stim_amp=65, noise_amp=-np.Inf,
                 output_dir='rawData', window_size=None, screen_num=None,
                 full_screen=True, force_quit=['escape', 'lctrl', 'rctrl'],
                 participant=None, session=None, verbose=None):

        # self._stim_fs = stim_fs
        self._stim_amp = stim_amp
        self._noise_amp = noise_amp
        self._force_quit = force_quit

        # some hardcoded parameters...
        bkgd_color = [-1, -1, -1]  # psychopy does RGB from -1 to 1
        root_dir = './'

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
        psylog.LogFile(basename + '.log', level=psylog.INFO)
        psylog.console.setLevel(psylog.WARNING)

        # clocks
        self.master_clock = core.Clock()
        self.trial_clock = core.Clock()

        # list of trial components
        self.trial_components = []

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
        else:
            raise TypeError('audio_controller must be a str or dict.')

        self.audio_type = audio_controller['TYPE'].lower()
        if self.audio_type == 'tdt':
            psylog.info('Expyfun: Setting up TDT')
            self.tdt = TDT(audio_controller)
            self.audio_type = self.tdt.model
            self._fs = self.tdt.fs
        elif self.audio_type == 'psychopy':
            psylog.info('Expyfun: Initializing PsychoPy audio')
            self.tdt = None
            self._fs = 44100
            self.audio = sound.Sound(np.zeros((1, 2)), sampleRate=self._fs)
            self.audio.setVolume(1)  # TODO: check this w/r/t stim_scaler
            #self.trial_components.append(self.audio)  # TODO: necessary?
        else:
            raise ValueError('audio_controller[\'TYPE\'] must be '
                             '\'psychopy\' or \'tdt\'.')

        # scaling factor to ensure uniform intensity across output devices
        self._stim_scaler = _get_stim_scaler(self.audio_type, stim_amp,
                                             stim_rms)

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

        # basic components
        self.data_handler = data.ExperimentHandler(name=exp_name, version='',
                                                   extraInfo=self.exp_info,
                                                   runtimeInfo=None,
                                                   originPath=None,
                                                   savePickle=True,
                                                   saveWideText=True,
                                                   dataFileName=basename)
        self.text_stim = visual.TextStim(win=self.win, text='', pos=[0, 0],
                                         height=0.1, wrapWidth=0.9,
                                         units='norm', color=[1, 1, 1],
                                         colorSpace='rgb', opacity=1.0,
                                         contrast=1.0, name='myTextStim',
                                         ori=0, depth=0, flipHoriz=False,
                                         flipVert=False, alignHoriz='center',
                                         alignVert='center', bold=False,
                                         italic=False, font='Arial',
                                         fontFiles=[], antialias=True)
        self.button_handler = event.BuilderKeyResponse()
        #self.shape_stim = visual.ShapeStim()

        # append to list of trial components
        self.trial_components.append(self.button_handler)
        self.trial_components.append(self.text_stim)
        #self.trial_components.append(self.shape_stim)

        self.master_clock.reset()
        psylog.info('Expyfun: Initialization complete')

    def __repr__(self):
        """Return a useful string representation of the experiment
        """
        string = ('<ExperimentController ({3}): "{0}" {1} ({2})>'
                  ''.format(self.exp_info['exp_name'],
                            self.exp_info['participant'],
                            self.exp_info['session'],
                            self.audio_type))
        return string

    def screen_prompt(self, text, max_wait=np.inf, min_wait=0, live_keys=[]):
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
        val : str | None
            The button that was pressed. Will be None if the function timed
            out before the subject responded.
        time : float
            The timestamp.
        """
        if np.isinf(max_wait) and live_keys is None:
            raise ValueError('You have asked for max_wait=inf with '
                             'live_keys=None, this will stall the experiment '
                             'forever.')
        self.screen_text(text)
        if live_keys is None:
            self.wait_secs(max_wait)
            return (None, max_wait)
        else:
            self.wait_secs(min_wait)
            return self.get_press(max_wait=max_wait - min_wait,
                                  live_keys=live_keys)

    def screen_text(self, text, clock=None):
        """Show some text on the screen. Wrapper for PsychoPy's
        visual.TextStim.SetText() method.

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
        self._flip()

    def clear_screen(self):
        """Remove all visual stimuli from the screen.
        """
        #self.text_stim.status = FINISHED
        for comp in self.trial_components:
            if hasattr(comp, 'setAutoDraw'):
                comp.setAutoDraw(False)
        self._flip()

    def init_trial(self):
        """Reset trial clock, clear keyboard buffer, reset trial components to
        default status.
        """
        self.trial_clock.reset()
        self.button_handler.keys = []
        self.button_handler.rt = []
        #for comp in self.trial_components:
        #    if hasattr(comp, 'status'):
        #        comp.status = NOT_STARTED

    def add_data_line(self, data_dict):
        """
        """
        for key, value in data_dict.items():
            self.data_handler.addData(key, value)
        self.data_handler.nextEntry()

    def get_press(self, max_wait=np.inf, min_wait=0.0, live_keys=[]):
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

        Returns
        -------
        Tuple (str, float) indicating the first key pressed and its timestamp.
        If no acceptable key is pressed between min_wait and max_wait, returns
        ([], None).
        """
        if self.response_device == 'keyboard':
            live_keys = map(str, live_keys)  # accept ints
            if len(self._force_quit):
                live_keys = live_keys + self._force_quit
            self.button_handler.keys = []
            self.button_handler.rt = []
            self.button_handler.clock.reset()
            self.wait_secs(min_wait)
            event.clearEvents('keyboard')
            pressed = []
            while (not len(pressed) and
                   self.button_handler.clock.getTime() < max_wait):
                pressed = event.getKeys(keyList=live_keys,
                                        timeStamped=self.button_handler.clock)
            if len(pressed):
                result = pressed[0]
                self._check_force_quit(result)
            else:
                result = ([], None)
            self.button_handler.keys = result[0]
            self.button_handler.rt = result[1]
            self.data_handler.addData('button_presses',
                                      self.button_handler.keys)
            self.data_handler.addData('reaction_times',
                                      self.button_handler.rt)
            self.data_handler.nextEntry()
            return result
        else:
            raise NotImplementedError()
            # TODO: check the proper tag name for our circuit
            # TODO: is there a way to detect proper tag name automatically?
            # self.tdt.rpcox.GetTagVal('ButtonBoxTagName')

    def get_presses(self, max_wait, min_wait=0.0, live_keys=[]):
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

        Returns
        -------
        List of tuples (str, float) indicating which keys were pressed and
        their timestamps.
        """
        assert min_wait < max_wait
        if self.response_device == 'keyboard':
            live_keys = map(str, live_keys)  # accept ints
            if len(self._force_quit):
                live_keys = live_keys + self._force_quit
            self.button_handler.keys = []
            self.button_handler.rt = []
            self.button_handler.clock.reset()
            self.wait_secs(min_wait)
            event.clearEvents('keyboard')
            pressed = []
            while self.button_handler.clock.getTime() < max_wait:
                pressed += event.getKeys(keyList=live_keys,
                                         timeStamped=self.button_handler.clock)
            if len(pressed):
                result = pressed
                self._check_force_quit([k for (k, t) in pressed])
            else:
                result = [([], None)]
            for (key, timestamp) in result:
                self.button_handler.keys = key
                self.button_handler.rt = timestamp
                self.data_handler.addData('button_presses',
                                          self.button_handler.keys)
                self.data_handler.addData('reaction_times',
                                          self.button_handler.rt)
                #self.data_handler.addData('trial', trial_num)
                self.data_handler.nextEntry()
            return result
        else:
            raise NotImplementedError()
            # TODO: check the proper tag name for our circuit
            # TODO: is there a way to detect proper tag name automatically?
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

    def wait_secs(self, secs, hog_cpu_time=0.2):
        """Wait a specified number of seconds.

        Parameters
        ----------
        secs : float
            Number of seconds to wait.
        hog_cpu_time : float
            Amount of CPU time to hog. See Notes.

        Notes
        -----
        From the PsychoPy documentation:
        If secs=10 and hogCPU=0.2 then for 9.8s python's time.sleep function
        will be used, which is not especially precise, but allows the cpu to
        perform housekeeping. In the final hogCPUperiod the more precise method
        of constantly polling the clock is used for greater precision.

        If you want to obtain key-presses during the wait, be sure to use
        pyglet and to hogCPU for the entire time, and then call
        psychopy.event.getKeys() after calling wait().

        If you want to suppress checking for pyglet events during the wait, do
        this once:
            core.checkPygletDuringWait = False
        and from then on you can do:
            core.wait(sec)
        This will preserve terminal-window focus during command line usage.
        """
        if any([secs < 0.2, secs < hog_cpu_time]):
            hog_cpu_time = secs
        core.wait(secs, hogCPUperiod=hog_cpu_time)

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
        psylog.info('Expyfun: Loading %d samples to buffer' % data.size)
        if self.tdt is not None:
            self.tdt.write_buffer(buffer_name, offset,
                                  data * self._stim_scaler)
        else:
            self.audio.setSound(np.asarray(data, order='C'))
            # TODO: check PsychoPy output w/r/t stim scaler
            #self.audio.setSound(np.asarray(data * self._stim_scaler,
            #                               order='C'))

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
            self._stop_tdt()
            self._reset_tdt()
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
        self._flip()
        self._play(clock)
        if self._fp_function is not None:
            self._fp_function()
        # TODO: note psychopy's self.win.callOnFlip(someFunction)

    def call_on_flip_and_play(self, function, *args, **kwargs):
        """Locus for additional functions to be executed on flip and play.
        """
        if function is not None:
            self._fp_function = partial(function, *args, **kwargs)
        else:
            self._fp_function = None
        # TODO: note psychopy's self.win.callOnFlip(someFunction)

    def set_noise_amp(self, new_amp):
        """TODO: add docstring
        """
        self._noise_amp = new_amp

    def set_stim_amp(self, new_amp):
        """TODO: add docstring
        """
        self._stim_amp = new_amp

    def _flip(self):
        """Flip the screen buffer.
        """
        self.win.flip()

    def _play(self, clock):
        """Play the audio buffer.
        """
        psylog.debug('Expyfun: playing audio')
        if self.tdt is not None:
            # TODO: detect which triggers are which rather than hard-coding
            self.tdt.trigger(1)
        else:
            self.audio.tStart = clock.getTime()
            self.audio.play()

    def _stop_tdt(self):
        """Stop TDT ring buffer playback.
        """
        psylog.debug('Stopping audio')
        # TODO: detect which triggers are which rather than hard-coding
        self.tdt.trigger(2)

    def _reset_tdt(self):
        """Reset TDT audio buffer to beginning.
        """
        psylog.debug('Expyfun: Resetting audio')
        # TODO: detect which triggers are which rather than hard-coding
        self.tdt.trigger(5)

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
        if self.tdt is not None:
            # TODO: detect which triggers are which rather than hard-coding
            self.tdt.trigger(4)  # kill noise
            self.stop_reset()
            self.tdt.halt_circuit()
        try:
            core.quit()
        except SystemExit:
            pass
        except Exception as e:
            print e
        if any([x is not None for x in (err_type, value, traceback)]):
            raise err_type, value, traceback

    @property
    def fs(self):
        """Playback frequency of the audio controller (samples / second).
        """
        return self._fs  # not user-settable


def _get_rms(audio_controller):
    if audio_controller == 'RM1':
        return 108  # this is approx; knob is not detented
    elif audio_controller == 'RP2':
        return 108
    elif audio_controller == 'RZ6':
        return 114
    elif audio_controller == 'psychopy':
        return 90  # TODO: this value not yet calibrated, may vary by system
    else:
        psylog.WARN('Unknown audio controller: stim scaler may not work '
                    'correctly. You may want to remove your headphones if this'
                    ' is the first run of your experiment.')
        return 90  # for untested TDT models


def _get_stim_scaler(audio_controller, stim_amp, stim_rms):
    exponent = (-(_get_rms(audio_controller) - stim_amp) / 20) / stim_rms
    return np.power(10, exponent)
