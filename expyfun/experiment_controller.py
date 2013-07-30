import logging, time
from functools import partial
from tdt.util import connect_rpcox, connect_zbus
from .utils import get_config, verbose

logger = logging.getLogger('expyfun')


class ExperimentController(object):
    @verbose
    def __init__(self, audio_controller=None, response_device=None,
                 verbose=None):
        """
        Interface for hardware control (audio, buttonbox, eye tracker, etc.)

        Parameters
        ----------
        audio_controller : str | None
            Can be 'psychopy' or a TDT model (e.g., 'RM1' or 'RP2'). If None,
            the type will be read from the system configuration file.

        response_device : str | None
            Can be 'keyboard' or 'buttonbox'.  If None, the type will be read
            from the system configuration file.

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

        if response_device is None:
            self.response_device = get_config('RESPONSE_DEVICE')
        else: 
            self.response_device = response_device

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
            self._fs = self.tdt.GetSFreq()
        
        # placeholder for extra actions to do on flip-and-play
        self._fp_function = None
        
        logger.info('Initialization complete')

    def load_buffer(self, data, buffer_name=None):
        """
        Method for loading audio data into the audio buffer.

        Parameters
        ----------
        data : np.array
            Audio data as floats scaled to (-1,+1), formatted as an Nx1 or Nx2
            numpy array with dtype float32.

        buffer_name : str | None
            Name of the TDT buffer to target. Ignored if audio_controller is 
            'psychopy'.

        Returns
        -------
        buffer_loaded : {0,1}
            Boolean integer indicating success or failure of buffer load.

        Notes
        -----
        Blah blah blah.
        """
        logger.info('Loading buffers with {} samples of data'.format(data.size))
        if not self.tdt is None:
            # load data into TDT buffer
            buffer_loaded = self.tdt.WriteTagV(buffer_name, 0, data)
        else:
            # TODO: add code to load data into psychopy buffer
            buffer_loaded = 0
        return buffer_loaded

    def clear_buffer(self, buffer_name=None):
        """ 
        Method for clearing audio data from the audio buffer.

        Parameters
        ----------
        buffer_name : str | None
            Name of the TDT buffer to target. Ignored if audio_controller is
            'psychopy'.

        Returns
        -------
        buffer_cleared : {0,1}
            Boolean integer indicating success or failure of buffer clear.

        Notes
        -----
        Blah blah blah.
        """
        logger.info('Clearing buffers')
        if not self.tdt is None:
            # clear data in TDT buffer
            buffer_cleared = self.tdt.ZeroTag(buffer_name)
        else:
            # TODO: add code to clear data from psychopy buffer
            buffer_cleared = 0
        return buffer_cleared

    def _play(self):
        # XXX should add a way to detect which triggers are which rather than
        # hard-coding
        logger.debug('playing')
        self.tdt.SoftTrg(1)

    def _stop(self):
        # XXX should add a way to detect which triggers are which rather than
        # hard-coding
        logger.debug('stopping')
        self.tdt.SoftTrg(2)

    def _reset(self):
        # XXX should add a way to detect which triggers are which rather than
        # hard-coding
        logger.debug('resetting')
        self.tdt.SoftTrg(5)

    def stop_reset(self):
        """
        XXX ADD DOCSTRING
        """
        logger.info('Stopping and resetting')
        self._stop()
        self._reset()

    def close(self):
        """
        XXX ADD DOCSTRING
        """
        self.__exit__()

    def __exit__(self, type, value, traceback):
        # XXX stop the TDT circuit, etc.  (for use with "with" syntax)
        logger.debug('exiting cleanly')
        if not self.tdt is None:
            # send stop triggers and halt circuit
            # XXX should add a way to detect which triggers are which rather
            # than hard-coding
            self.tdt.SoftTrg(4) # kill noise
            self.stop_reset()
            self.tdt.Halt()

    def __enter__(self):
        # wrap to init? may want to do some low-level stuff to make sure the
        # connection is working?
        # XXX for with statement (for use with "with" syntax)
        logger.debug('Entering')
        return self

    def flip_and_play(self):
        # XXX flip screen
        # XXX play buffer
        logger.info('Flipping and playing audio')
        if self._fp_function is not None:
            self._fp_function()

    def call_on_flip_and_play(self, function, *args, **kwargs):
        if function is not None:
            self._fp_function = partial(function, *args, **kwargs)
        else:
            self._fp_function = None

    @property
    def fs(self):
        # do it this way so people can't set it
        return self._fs


class TDTObject(object):
    def __init__(self, tdt_type, circuit, interface):
        """
        Interface for audio output.

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
        
        """
        # HIGH-LEVEL APPROACH
        # (fails often, possibly due to inappropriate use of zBUS)
        self.rpcox = tdt.DSPCircuit(self.circuit, self.audio_controller,
                                  interface=self.tdt_interface)
        self.tdt.start()

        # LOW-LEVEL APPROACH (works reliably, no device abstraction)
        self.rpcox = tdt.actxobjects.RPcoX()
        self.connection = self.rpcox.ConnectRM1(IntName=self.tdt_interface,
                                                DevNum=1)
        """
        # MID-LEVEL APPROACH
        self.rpcox = connect_rpcox(name=self.audio_controller,
                                   interface=self.tdt_interface,
                                   device_id=1, address=None)
        if not self.rpcox is None:
            logger.info('RPcoX connection established.')
        else:
            raise ExperimentError('Problem initializing RPcoX.')

        """
        # May need to implement zBUS for devices other than RM1
        self.zbus = connect_zbus(interface=self.tdt_interface)
        if not self.zbus is None:
            logger.info('zBUS connection established.')
        else:
            raise ExperimentError('Problem initializing zBUS.')
        """

        if self.rpcox.LoadCOF(self.circuit):
            logger.info('Circuit loaded.')
        else:
            try:
                if self.rpcox.ClearCOF(): 
                    logger.info('Circuit cleared.')
                time.sleep(0.25)
                if self.rpcox.LoadCOF(self.circuit): 
                    logger.info('Circuit loaded.')
            except:
                raise ExperimentError('Problem loading circuit.')

        print('Circuit {} loaded to {} via {}.'.format(self.circuit,
                    self.audio_controller, self.tdt_interface))

        if self.tdt.Run(): logger.info('Circuit running.')
        else: raise ExperimentError('Problem starting circuit.')

        time.sleep(0.25)
        self._fs = self.tdt.GetSFreq()
        
    def trigger(self, trigger_number):
        """
        Wrapper for "SoftTrg"

        Parameters
        ----------
        trigger_number : int
            Trigger number to send to TDT.

        Returns
        -------
        trigger_sent : {0,1}
            Boolean integer indicating success or failure of buffer clear.

        Notes
        -----
        Blah blah blah.
        """
        self.SoftTrg(trigger_number)


class ExperimentError(Exception):
    """ 
    Exceptions unique to the ExperimentController class and its derivatives.

    Attributes:
        msg -- explanation of the error.
    """

    def __init__(self, msg):
        self.msg = msg
