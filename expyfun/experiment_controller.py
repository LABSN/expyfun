"""TODO: add docstring
"""
import logging
import time
import os
import numpy as np
#from os.path import sep  # used in basename
from functools import partial
from tdt.util import connect_rpcox, connect_zbus
from psychopy import visual, core, data, event, sound, gui
from .utils import get_config, verbose

logger = logging.getLogger('expyfun')


class ExperimentController(object):
    """TODO: add docstring
    """
    @verbose
    def __init__(self, exp_name, audio_controller=None, response_device=None,
                 stim_rms=0.01, stim_ampl=65, noise_ampl=-np.inf,
                 verbose=None):
        """Interface for hardware control (audio, buttonbox, eye tracker, etc.)

        Parameters
        ----------
        exp_name : str
            Name of the experiment.
        audio_controller : str | None
            Can be 'psychopy' or a TDT model (e.g., 'RM1' or 'RP2'). If None,
            the type will be read from the system configuration file.
        response_device : str | None
            Can be 'keyboard' or 'buttonbox'.  If None, the type will be read
            from the system configuration file.
        stim_rms : float
            The RMS amplitude of the stimuli (strongly recommended to be 0.01).
        stim_ampl : float
            The desired dB SPL at which to play the stimuli.
        noise_ampl : float
            The desired dB SPL at which to play the dichotic noise.
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

        # TODO: okay to leave this stuff hardcoded?
        #exp_root = './'  # used in basename
        output_dir = 'rawData'
        bkgd_color = [-1, -1, -1]  # psychopy does RGB from -1 to 1
        """XXX ORPHANED DOCSTRING FROM MOVE TO HARDCODING
        output_dir : str | 'rawData'
            An absolute or relative path to a directory in which raw experiment
            data will be stored. If output_folder does not exist, it will be
            created.
        """

        # dictionary for experiment metadata
        self.exp_info = {'participant': '', 'session': '001',
                         'exp_name': exp_name, 'date': data.getDateStr()}

        # session start dialog
        session_dialog = gui.DlgFromDict(dictionary=self.exp_info,
                                         fixed=['exp_name', 'date'],
                                         title=exp_name)
        if session_dialog == False:
            core.quit()  # user pressed cancel

        # initialize output folder for raw data
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        """
        basename = exp_root + sep + output_dir + sep + '{0}_{1}'.format(
                   self.exp_info['participant'], self.exp_info['date'])
        log_file = logging.LogFile(basename + '.log', level=logging.EXP)
        logging.console.setLevel(logging.WARNING)
        exp_handler = data.ExperimentHandler(name=exp_name, version='',
                                             extraInfo=expInfo,
                                             runtimeInfo=None, originPath=None,
                                             savePickle=True,
                                             saveWideText=True,
                                             dataFileName=basename)
        """

        # create visual window
        self.win = visual.Window(size=get_config('WINDOW_SIZE'), fullscr=True,
                                 screen=0, allowGUI=False, allowStencil=False,
                                 monitor='testMonitor', color=bkgd_color,
                                 colorSpace='rgb')

        # response device
        if response_device is None:
            self.response_device = get_config('RESPONSE_DEVICE')
        else:
            self.response_device = response_device

        # audio setup
        if audio_controller is None:
            self.audio_controller = get_config('AUDIO_CONTROLLER')
        else:
            self.audio_controller = audio_controller
        if self.audio_controller is 'psychopy':
            self.tdt = None
            self._fs = 22050
        else:
            self.tdt = TDTObject(self.audio_controller,
                                 get_config('TDT_CIRCUIT'),
                                 get_config('TDT_INTERFACE'))
            self._fs = self.tdt.fs

        # TODO: add function _get_stim_scaler to do the calculations
        self._stim_scaler = _get_stim_scaler(self.audio_controller, stim_ampl,
                                            stim_rms)

        # placeholder for extra actions to do on flip-and-play
        self._fp_function = None

        logger.info('Initialization complete')

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
        logger.info('Loading buffers with {} samples...'.format(data.size))
        if not self.tdt is None:
            self.tdt.write_buffer(buffer_name, offset,
                                  data * self._stim_scaler)
        else:
            # TODO: psychopy write sound buffer method
            raise NotImplementedError()

    def clear_buffer(self, buffer_name=None):
        """Clear audio data from the audio buffer.

        Parameters
        ----------
        buffer_name : str | None
            Name of the TDT buffer to target. Ignored if audio_controller is
            'psychopy'.
        """
        logger.info('Clearing buffer...')
        if not self.tdt is None:
            self.tdt.clear_buffer(buffer_name)
        else:
            # TODO: psychopy clear sound buffer method
            raise NotImplementedError()

    def _play(self):
        """Play the audio buffer.
        """
        logger.debug('playing...')
        if not self.tdt is None:
            # TODO: detect which triggers are which rather than hard-coding
            self.tdt.trigger(1)
        else:
            # TODO: psychopy play sound method
            raise NotImplementedError()

    def _stop(self):
        """Stop audio buffer playback.
        """
        logger.debug('stopping...')
        if not self.tdt is None:
            # TODO: detect which triggers are which rather than hard-coding
            self.tdt.trigger(2)
        else:
            # TODO: psychopy stop sound method
            raise NotImplementedError()

    def _reset(self):
        """Reset audio buffer to beginning.
        """
        logger.debug('resetting')
        if not self.tdt is None:
            # TODO: detect which triggers are which rather than hard-coding
            self.tdt.trigger(5)
        else:
            # TODO: psychopy reset sound buffer method
            raise NotImplementedError()

    def stop_reset(self):
        """Stop audio buffer playback and reset cursor to beginning of buffer.
        """
        logger.info('Stopping and resetting')
        self._stop()
        self._reset()

    def close(self):
        """Close all connections in experiment controller.
        """
        self.__exit__(None, None, None)

    def __exit__(self, type, value, traceback):
        """
        Notes
        -----
        type, value and traceback will be None when called by self.close()
        """
        # stop the TDT circuit, etc.  (for use with "with" syntax)
        logger.debug('exiting cleanly')
        if not self.tdt is None:
            # TODO: detect which triggers are which rather than hard-coding
            self.tdt.trigger(4)  # kill noise
            self.stop_reset()
            self.tdt.halt_circuit()
        else:
            # TODO: psychopy exit method
            raise NotImplementedError()

    def __enter__(self):
        # (for use with "with" syntax) wrap to init? may want to do some
        # low-level stuff to make sure the connection is working?
        logger.debug('Entering')
        return self

    def flip_and_play(self):
        """Flip screen and immediately begin playing audio.
        """
        logger.info('Flipping and playing audio')
        # TODO: flip screen
        self._play()
        if self._fp_function is not None:
            self._fp_function()

    def call_on_flip_and_play(self, function, *args, **kwargs):
        """Locus for additional functions to be executed on flip and play.
        """
        if function is not None:
            self._fp_function = partial(function, *args, **kwargs)
        else:
            self._fp_function = None

    def set_noise_ampl(self):
        """TODO: add docstring
        """
        # TODO: look at MATLAB getStimScaler and TDT.noiseamp
        raise NotImplementedError()

    def set_stim_ampl(self):
        """TODO: add docstring
        """
        # TODO: look at MATLAB getStimScaler
        # set property self._stim_scaler
        raise NotImplementedError()

    @property
    def fs(self):
        """Playback frequency of the audio controller (samples / second).
        """
        # do it this way so people can't set it
        return self._fs


class TDTObject(object):
    """ TODO: add docstring
    """
    def __init__(self, tdt_type, circuit, interface):
        """Interface for audio output.

        Parameters
        ----------
        tdt_type : str
            String name of the TDT model (e.g., 'RM1', 'RP2', etc).
        circuit : str
            Path to the TDT circuit.
        interface : {'USB','GB'}
            Type of interface between computer and TDT (USB or Gigabit).

        Returns
        -------
        tdt_obj : instance of a TDTObject.
            The object containing all relevant info about the TDT in use.

        Notes
        -----
        Blah blah blah.
        """
        self.circuit = circuit
        self.tdt_type = tdt_type
        self.interface = interface

        # initialize RPcoX connection
        """
        # HIGH-LEVEL APPROACH
        # (fails often, possibly due to inappropriate zBUS call in DSPCircuit)
        import tdt
        self.rpcox = tdt.DSPCircuit(circuit, tdt_type, interface=interface)
        self.rpcox.start()

        # LOW-LEVEL APPROACH (works reliably, but no device abstraction)
        self.rpcox = tdt.actxobjects.RPcoX()
        self.connection = self.rpcox.ConnectRM1(IntName=interface, DevNum=1)
        """
        # MID-LEVEL APPROACH
        self.rpcox = connect_rpcox(name=tdt_type, interface=interface,
                                   device_id=1, address=None)
        if not self.rpcox is None:
            logger.info('RPcoX connection established.')
        else:
            raise ExperimentError('Problem initializing RPcoX.')

        """
        # start zBUS (may be needed for devices other than RM1)
        self.zbus = connect_zbus(interface=interface)
        if not self.zbus is None:
            logger.info('zBUS connection established.')
        else:
            raise ExperimentError('Problem initializing zBUS.')
        """

        # load circuit
        if self.rpcox.LoadCOF(circuit):
            logger.info('Circuit loaded.')
        else:
            logger.debug('Problem loading circuit. Trying to clear first...')
            try:
                if self.rpcox.ClearCOF():
                    logger.debug('Circuit cleared.')
                time.sleep(0.25)
                if self.rpcox.LoadCOF(circuit):
                    logger.info('Circuit loaded.')
            except:
                raise ExperimentError('Problem loading circuit.')
        logger.info('Circuit {} loaded to {} via {}.'.format(circuit,
                    tdt_type, interface))

        # run circuit
        if self.rpcox.Run():
            logger.info('Circuit running.')
        else:
            raise ExperimentError('Problem starting circuit.')
        time.sleep(0.25)

    @property
    def fs(self):
        """Playback frequency of the TDT circuit (samples / second).
        """
        # read sampling rate from circuit
        return self.rpcox.GetSFreq()

    def trigger(self, trigger_number):
        """Wrapper for tdt.util.RPcoX.SoftTrg()

        Parameters
        ----------
        trigger_number : int
            Trigger number to send to TDT.

        Returns
        -------
        trigger_sent : {0,1}
            Boolean integer indicating success or failure of buffer clear.
        """
        self.rpcox.SoftTrg(trigger_number)

    def write_buffer(self, data, offset, buffer_name):
        """Wrapper for tdt.util.RPcoX.WriteTagV()
        """
        # TODO: check to make sure data is properly formatted / correct dtype
        self.rpcox.WriteTagV(buffer_name, offset, data)

    def clear_buffer(self, buffer_name):
        """Wrapper for tdt.util.RPcoX.ZeroTag()
        """
        self.rpcox.ZeroTag(buffer_name)

    def halt_circuit(self):
        """Wrapper for tdt.util.RPcoX.Halt()
        """
        self.rpcox.Halt()


def _get_stim_scaler(audio_controller, stim_ampl, stim_rms):
    return 10 ^ (-(_get_tdt_rms(audio_controller) - stim_ampl) / 20) / stim_rms


def _get_tdt_rms(tdt_type):
    if tdt_type is 'RM1':
        return 108  # this is approx; knob is not detented
    elif tdt_type is 'RP2':
        return 108
    elif tdt_type is 'RZ6':
        return 114
    else:
        return 90  # for sound cards


def get_tdt_rates():
    return [6103.515625, 12207.03125, 24414.0625, 48828.125, 97656.25,
            195312.5]


class ExperimentError(Exception):
    """
    Exceptions unique to the ExperimentController class and its derivatives.

    Attributes:
        msg -- explanation of the error.
    """

    def __init__(self, msg):
        self.msg = msg