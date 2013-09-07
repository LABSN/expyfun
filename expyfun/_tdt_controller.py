import time
import numpy as np
import platform
from os import path as op
if 'Windows' in platform.platform():
    from tdt.util import connect_rpcox, connect_zbus
else:
    connect_rpcox, connect_zbus = None, None
from psychopy import logging as psylog

from .utils import get_config, wait_secs
from ._input_controllers import BaseKeyboard


class TDTController(BaseKeyboard):
    """Interface for TDT audio output, stamping, and responses

    Parameters
    ----------
    tdt_params : dict | None
        A dictionary containing keys:
        'TYPE' (this should always be 'tdt');
        'TDT_MODEL' (String name of the TDT model ('RM1', 'RP2', etc));
        'TDT_CIRCUIT_PATH' (Path to the TDT circuit); and
        'TDT_INTERFACE' (Type of connection, either 'USB' or 'GB').
    ec : instance of ExperimentController
        The parent EC.
    force_quit_keys : list | None | bool
        Keys to use to quit when initializing in keyboard mode. If False,
        don't bother initializing as a keyboard.

    Returns
    -------
    tdt_obj : instance of a TDTObject.
        The object containing all relevant info about the TDT in use.
    """
    def __init__(self, tdt_params, ec, force_quit_keys):
        legal_keys = ['TYPE', 'TDT_MODEL', 'TDT_CIRCUIT_PATH', 'TDT_INTERFACE']
        if tdt_params is None:
            tdt_params = {'TYPE': 'tdt'}
        if not isinstance(tdt_params, dict):
            raise TypeError('tdt_params must be a dictionary.')
        for k in legal_keys:
            if k not in tdt_params.keys() and k != 'TYPE':
                tdt_params[k] = get_config(k)
        for k in tdt_params.keys():
            if k not in legal_keys:
                raise KeyError('Unrecognized key in tdt_params: {0}'.format(k))
        self._model = tdt_params['TDT_MODEL']

        if tdt_params['TDT_CIRCUIT_PATH'] is None:
            cl = dict(RM1='RM1', RP2='RM1', RZ6='RZ6')
            self._circuit = op.join(op.split(__file__)[0], 'tdt-circuits',
                                    'expCircuitF32_' + cl[self._model] +
                                    '.rcx')
        else:
            self._circuit = tdt_params['TDT_CIRCUIT_PATH']
        if not op.isfile(self._circuit):
            raise IOError('Could not find file {}'.format(self._circuit))
        if tdt_params['TDT_INTERFACE'] is None:
            tdt_params['TDT_INTERFACE'] = 'USB'
        self._interface = tdt_params['TDT_INTERFACE']

        # initialize RPcoX connection
        """
        # HIGH-LEVEL APPROACH, fails possibly due to zBUS call in DSPCircuit
        self.rpcox = tdt.DSPCircuit(circuit, tdt_type, interface=interface)
        self.rpcox.start()

        # LOW-LEVEL APPROACH (works reliably, but no device abstraction)
        self.rpcox = tdt.actxobjects.RPcoX()
        self.connection = self.rpcox.ConnectRM1(IntName=interface, DevNum=1)
        """
        # MID-LEVEL APPROACH
        if connect_rpcox is not None:
            try:
                self.rpcox = connect_rpcox(name=self.model,
                                           interface=self.interface,
                                           device_id=1, address=None)
            except Exception as exp:
                raise OSError('Could not connect to {}, is it turned on? '
                              '(TDT message: "{}")'.format(self._model, exp))

            if self.rpcox is not None:
                psylog.info('Expyfun: RPcoX connection established')
            else:
                raise IOError('Problem initializing RPcoX.')
            """
            # start zBUS (may be needed for devices other than RM1)
            self.zbus = connect_zbus(interface=interface)
            if self.zbus is not None:
                psylog.info('Expyfun: zBUS connection established')
            else:
                raise ExperimentError('Problem initializing zBUS.')
            """
            # load circuit
            if self.rpcox.LoadCOF(self.circuit):
                psylog.info('Expyfun: TDT circuit loaded')
            else:
                psylog.debug('Expyfun: Problem loading circuit. Clearing...')
                try:
                    if self.rpcox.ClearCOF():
                        psylog.debug('Expyfun: TDT circuit cleared')
                    time.sleep(0.25)
                    if self.rpcox.LoadCOF(self.circuit):
                        psylog.info('Expyfun: TDT circuit loaded')
                except:
                    raise IOError('Expyfun: Problem loading circuit.')
            psylog.info('Expyfun: Circuit {0} loaded to {1} via {2}.'
                        ''.format(self.circuit, self.model, self.interface))
            # run circuit
            if self.rpcox.Run():
                psylog.info('Expyfun: TDT circuit running')
            else:
                raise SystemError('Expyfun: Problem starting TDT circuit.')
            time.sleep(0.25)

        self.clear_buffer()

        # do BaseKeyboard init last, to make sure circuit is running
        if force_quit_keys is not False:
            BaseKeyboard.__init__(self, ec, force_quit_keys)

################################ AUDIO METHODS ###############################
    def load_buffer(self, data):
        """Load audio samples into TDT buffer.

        Parameters
        ----------
        data : np.array
            Audio data as floats scaled to (-1,+1), formatted as an Nx2 numpy
            array with dtype 'float32'.
        """
        self.rpcox.WriteTagV('datainleft', 0, data[:, 0])
        self.rpcox.WriteTagV('datainright', 0, data[:, 1])

    def clear_buffer(self):
        """Clear the TDT ring buffers.
        """
        self.rpcox.ZeroTag('datainleft')
        self.rpcox.ZeroTag('datainright')

    def play(self):
        """Send the soft trigger to start the ring buffer playback.
        """
        self._trigger(1)
        psylog.debug('Expyfun: Starting TDT ring buffer')

    def stop(self):
        """Stop playback and reset the buffer position"""
        self.pause()
        self.reset()

    def pause(self):
        """Send the soft trigger to stop the ring buffer playback.
        """
        self._trigger(2)
        psylog.debug('Stopping TDT audio')

    def start_noise(self):
        """Send the soft trigger to start the noise generator.
        """
        self._trigger(3)
        psylog.debug('Expyfun: Starting TDT noise')

    def stop_noise(self):
        """Send the soft trigger to stop the noise generator.
        """
        self._trigger(4)
        psylog.debug('Expyfun: Stopping TDT noise')

    def set_noise_level(self, new_level):
        """Set the amplitude of stationary background noise.
        """
        self.rpcox.SetTagVal('noiselev', new_level)

    def reset(self):
        """Send the soft trigger to reset the ring buffer to start position.
        """
        self._trigger(5)
        psylog.debug('Expyfun: Resetting TDT ring buffer')

################################ TRIGGER METHODS #############################
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
            self.rpcox.SetTagVal('trgname', trig)
            self._trigger(6)
            if ti < len(triggers):
                wait_secs(delay)

    def _trigger(self, trig):
        """Wrapper for tdt.util.RPcoX.SoftTrg()

        Parameters
        ----------
        trigger_number : int
            Trigger number to send to TDT.
        """
        if not self.rpcox.SoftTrg(trig):
            psylog.warn('SoftTrg failure for trigger: {}'.format(trig))

################################ KEYBOARD METHODS ############################

    def _get_keyboard_timebase(self):
        """Return time since circuit was started (in seconds).
        """
        return self.rpcox.GetTagVal('masterclock') / float(self.fs)

    def _clear_events(self):
        """Clear keyboard buffers.
        """
        self._trigger(7)

    def _retrieve_events(self, live_keys):
        """Values and timestamps currently in keyboard buffer.
        """
        press_count = int(round(self.rpcox.GetTagVal('npressabs')))
        if press_count > 0:
            # this one is indexed from zero
            press_times = self.rpcox.ReadTagVEX('presstimesabs', 0,
                                                press_count, 'I32', 'I32', 1)
            # this one is indexed from one (silly)
            press_vals = self.rpcox.ReadTagVEX('pressvalsabs', 1, press_count,
                                               'I32', 'I32', 1)
            press_times = np.array(press_times[0], float) / self.fs
            press_vals = np.log2(np.array(press_vals[0], float)) + 1
            press_vals = [str(int(round(p))) for p in press_vals]
            return [(v, t) for v, t in zip(press_vals, press_times)]
        else:
            return []

    def halt(self):
        """Wrapper for tdt.util.RPcoX.Halt()."""
        self.rpcox.Halt()
        psylog.debug('Expyfun: Halting TDT circuit')

############################# READ-ONLY PROPERTIES ###########################
    @property
    def fs(self):
        """Playback frequency of the audio (samples / second)."""
        return np.float(self.rpcox.GetSFreq())

    @property
    def model(self):
        """String representation of TDT model name ('RM1', 'RP2', etc)."""
        return self._model

    @property
    def circuit(self):
        """TDT circuit path."""
        return self._circuit

    @property
    def interface(self):
        """String representation of TDT interface ('USB' or 'GB')."""
        return self._interface


def get_tdt_rates():
    return {'6k': 6103.515625, '12k': 12207.03125, '25k': 24414.0625,
            '50k': 48828.125, '100k': 97656.25, '200k': 195312.5}
