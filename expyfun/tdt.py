import time
import numpy as np
import platform
if platform.platform == 'Windows':
    from tdt.util import connect_rpcox, connect_zbus
else:
    connect_rpcox = None
    connect_zbus = None
from psychopy import logging as psylog
from .utils import get_config, wait_secs


class TDT(object):
    """ TODO: add docstring
    """
    def __init__(self, tdt_params):
        """Interface for audio output.

        Parameters
        ----------
        tdt_params : dict | None
            A dictionary containing keys:
            'TYPE' (this should always be 'tdt');
            'TDT_MODEL' (String name of the TDT model ('RM1', 'RP2', etc));
            'TDT_CIRCUIT_PATH' (Path to the TDT circuit); and
            'TDT_INTERFACE' (Type of connection, either 'USB' or 'GB').

        Returns
        -------
        tdt_obj : instance of a TDTObject.
            The object containing all relevant info about the TDT in use.
        """
        # validate / populate parameters
        # TODO: add test for this
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
        self._circuit = tdt_params['TDT_CIRCUIT_PATH']
        self._interface = tdt_params['TDT_INTERFACE']

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
        if not connect_rpcox is None:
            self.rpcox = connect_rpcox(name=self.model,
                                       interface=self.interface,
                                       device_id=1, address=None)
            if not self.rpcox is None:
                psylog.info('Expyfun: RPcoX connection established')
            else:
                raise IOError('Problem initializing RPcoX.')
            """
            # start zBUS (may be needed for devices other than RM1)
            self.zbus = connect_zbus(interface=interface)
            if not self.zbus is None:
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

    @property
    def fs(self):
        """Playback frequency of the audio (samples / second).

        Notes
        -----
        When using TDT for audio, fs is read from the TDT circuit definition.
        """
        return np.float(self.rpcox.GetSFreq())

    @property
    def model(self):
        """String representation of TDT model name ('RM1', 'RP2', etc).
        """
        return self._model

    @property
    def circuit(self):
        """TDT circuit path.
        """
        return self._circuit

    @property
    def interface(self):
        """String representation of TDT interface ('USB' or 'GB').
        """
        return self._interface

    def _trigger(self, trigger_number):
        """Wrapper for tdt.util.RPcoX.SoftTrg()

        Parameters
        ----------
        trigger_number : int
            Trigger number to send to TDT.
        """
        # TODO: do we want to return the 0 or 1 passed by SoftTrg() ?
        self.rpcox.SoftTrg(trigger_number)

    def write_buffer(self, data, buffer_name, offset=0):
        """Wrapper for tdt.util.RPcoX.WriteTagV()
        """
        # TODO: check to make sure data is properly formatted / correct dtype
        # check dimensions of array
        # cast as np.float32 with order='C'
        # handle left/right channels
        self.rpcox.WriteTagV(buffer_name, offset, data)

    def clear_buffer(self, buffer_name):
        """Wrapper for tdt.util.RPcoX.ZeroTag()
        """
        # TODO: handle left/right channels
        self.rpcox.ZeroTag(buffer_name)

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
            self.trigger(6)
            if ti < len(triggers):
                wait_secs(delay)

    def halt_circuit(self):
        """Wrapper for tdt.util.RPcoX.Halt()
        """
        self.rpcox.Halt()

    def play(self):
        """Send the soft trigger to start the ring buffer playback.
        """
        psylog.debug('Expyfun: Starting TDT ring buffer')
        self._trigger(1)

    def stop(self):
        """Send the soft trigger to stop the ring buffer playback.
        """
        psylog.debug('Stopping TDT audio')
        self._trigger(2)

    def play_noise(self):
        """Send the soft trigger to start the noise generator.
        """
        psylog.debug('Expyfun: Starting TDT noise')
        self._trigger(3)

    def stop_noise(self):
        """Send the soft trigger to stop the noise generator.
        """
        psylog.debug('Expyfun: Stopping TDT noise')
        self._trigger(4)

    def reset(self):
        """Send the soft trigger to reset the ring buffer.
        """
        psylog.debug('Expyfun: Resetting TDT ring buffer')
        self._trigger(5)

    def get_first_press(self, max_wait, min_wait, live_buttons):
        """Returns only the first button pressed after min_wait.

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
        presses : list
            A list of tuples (str, float) indicating the key(s) pressed and
            their timestamp(s). If no acceptable keys were pressed between
            min_wait and max_wait, returns the one-item list [([], None)].
        """
        raise NotImplementedError()
        # TODO: implement

    def get_presses(self, max_wait, min_wait, live_buttons):
        """Get presses from min_wait to max_wait, from buttons in live_buttons.

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
        presses : list
            A list of tuples (str, float) indicating the key(s) pressed and
            their timestamp(s). If no acceptable keys were pressed between
            min_wait and max_wait, returns the one-item list [([], None)].
        """
        raise NotImplementedError()
        # TODO: implement
        """
        press_count = self.tdt.rpcox.GetTagVal('pressind')
        # Name, nOS, nWords, SrcType, DstType, nChans
        press_times = self.tdt.rpcox.ReadTagVEX('presstimes', 1,
                                                press_count, 'I32', 'F64',
                                                1) / float(self.fs)
        press_vals = np.ones_like(press_times)
        return zip(press_times, press_vals)
        """
        #MATLAB getPresses.m
        #WaitSecs(respWindow);
        #pressCount = invoke(TDT.RP, 'GetTagVal', 'pressind');
        #pressTimes = invoke(TDT.RP, 'ReadTagVEX', 'presstimes', 1, pressCount, 'I32', 'F64', 1)/TDT.fs;
        #pressVals = ones(size(pressTimes)); % TDT circuit should be upgraded to log all presses, rather than just 1-button
        #pressTimes = pressTimes - tSound;
        #pressTimes([false; diff(pressTimes)<0.1]) = []; % De-bounce presses
        #pressTimes = pressTimes((pressTimes >= 0) & (pressTimes <= respWindow));

        #MATLAB ReturnButtonNum.m
        #resp_num = buttonbox2num( invoke(TDT.RP, 'GetTagVal', 'currentbboxval') );

        #MATLAB buttonBox2Num.m
        #unique_button_codes = 2.^(0:7);
        #resp_num = find(unique_button_codes == button_box_code);
        #else resp_num = -1; % double press or no press


def get_tdt_rates():
    return {'6k': 6103.515625, '12k': 12207.03125, '25k': 24414.0625,
            '50k': 48828.125, '100k': 97656.25, '200k': 195312.5}
