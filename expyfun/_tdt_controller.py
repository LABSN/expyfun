"""Hardware interfaces for TDT RP2(.1), RM1, and RZ6."""

# Authors: Dan McCloy <drmccloy@uw.edu>
#          Eric Larson <larsoner@uw.edu>
#          Ross Maddox <rkmaddox@uw.edu>
#
# License: BSD (3-clause)

import time
import numpy as np
from os import path as op
from functools import partial
from copy import deepcopy
import warnings

from ._utils import get_config, wait_secs, logger, ZeroClock
from ._input_controllers import Keyboard


def _dummy_fun(self, name, ret, *args, **kwargs):
    logger.info('dummy-tdt: {0} {1}'.format(name, str(args)[:20] + ' ... ' +
                                            str(kwargs)[:20] + ' ...'))
    return ret


class DummyRPcoX(object):
    def __init__(self, model, interface):
        self.model = model
        self.interface = interface
        names = ['LoadCOF', 'ClearCOF', 'Run', 'ZeroTag', 'SetTagVal',
                 'GetSFreq', 'Halt']
        returns = [True, True, True, True, True,
                   24414.0125, True]
        for name, ret in zip(names, returns):
            setattr(self, name, partial(_dummy_fun, self, name, ret))
        self._clock = ZeroClock()
        self._stim_dur = 0
        self._play_start = 0

    def WriteTagVEX(self, name, offset, kind, data):
        if name == 'datainleft':
            self._stim_dur = len(data) / self.GetSFreq()
        return True

    def SoftTrg(self, trignum):
        if trignum == 1:
            self._play_start = time.time()
        elif trignum == 2:
            self._play_start -= self._stim_dur
        return True

    def GetTagVal(self, name):
        if name == 'masterclock':
            return self._clock.get_time()
        elif name == 'npressabs':
            return 0
        elif name == 'playing':
            return (time.time() - self._play_start < self._stim_dur)
        else:
            raise ValueError('unknown tag "{0}"'.format(name))


class TDTController(Keyboard):
    """Interface for TDT audio output, stamping, and responses

    .. warning:: This class should not be instantiated manually,
                 but rather should be created automatically by an
                 appropriate call to :class:`ExperimentController`.

    Parameters
    ----------
    tdt_params : dict | None
        A dictionary containing keys with string values:

            * 'TYPE': this should always be 'tdt'.
            * 'TDT_MODEL': String name of the TDT model, can be 'RM1', 'RP2',
              'RP2legacy', 'RZ6', or 'dummy' (default). For historical
              reasons, 'RP2' corresponds to the RP2.1, and 'RP2legacy'
              corresponds to the first-revision RP2.
            * 'TDT_CIRCUIT_PATH': Path to the TDT circuit. Defaults to an
              internal expyfun circuit.
            * 'TDT_INTERFACE': Type of connection, either 'USB' (default)
              or 'GB').
            * 'TDT_DELAY': the delay (in ms) for the circuit (default: '0').
            * 'TDT_TRIG_DELAY': additional delay for the triggers
              (default: '0').

        Note that the defaults are overridden on individual machines by
        the configuration file.

    Returns
    -------
    tdt_obj : instance of a TDTObject.
        The object containing all relevant info about the TDT in use.
    """
    def __init__(self, tdt_params):
        legal_keys = ['TYPE', 'TDT_MODEL', 'TDT_CIRCUIT_PATH', 'TDT_INTERFACE',
                      'TDT_DELAY', 'TDT_TRIG_DELAY']
        tdt_params = dict(TYPE='tdt') if tdt_params is None else tdt_params
        tdt_params = deepcopy(tdt_params)
        if not isinstance(tdt_params, dict):
            raise TypeError('tdt_params must be a dict, got '
                            '{0}'.format(type(tdt_params)))
        if tdt_params['TYPE'] != 'tdt':
            raise ValueError('tdt_params["TYPE"] must be "tdt", not '
                             '{0}'.format(tdt_params['TYPE']))
        # Set sensible defaults for values that are not passed
        defaults = dict(TDT_MODEL='dummy', TDT_DELAY='0', TDT_TRIG_DELAY='0',
                        TYPE='tdt')  # if not listed -> None
        for k in legal_keys:
            tdt_params[k] = tdt_params.get(
                k, get_config(k, defaults.get(k, None)))
        for key in ('TDT_DELAY', 'TDT_TRIG_DELAY'):
            tdt_params[key] = int(tdt_params[key])
        if tdt_params['TDT_DELAY'] < 0:
            raise ValueError('tdt_delay must be non-negative.')

        # Check keys
        for k in tdt_params.keys():
            if k not in legal_keys:
                raise KeyError('Unrecognized key in tdt_params {0}, must be '
                               'one of {1}'.format(k, ', '.join(legal_keys)))
        self._model = tdt_params['TDT_MODEL']
        legal_models = ['RM1', 'RP2', 'RZ6', 'dummy']
        if self._model not in legal_models:
            raise ValueError('TDT_MODEL="{0}" must be one of '
                             '{1}'.format(self._model, legal_models))

        if tdt_params['TDT_CIRCUIT_PATH'] is None and self._model != 'dummy':
            cl = dict(RM1='RM1', RP2='RM1', RP2legacy='RP2legacy', RZ6='RZ6')
            self._circuit = op.join(op.dirname(__file__), 'data',
                                    'expCircuitF32_' + cl[self._model] +
                                    '.rcx')
        else:
            self._circuit = tdt_params['TDT_CIRCUIT_PATH']
        if self._model != 'dummy' and not op.isfile(self._circuit):
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
        if tdt_params['TDT_MODEL'] != 'dummy':
            from tdt.util import connect_rpcox
            try:
                self.rpcox = connect_rpcox(name=self.model,
                                           interface=self.interface,
                                           device_id=1, address=None)
            except Exception as exp:
                raise OSError('Could not connect to {}, is it turned on? '
                              '(TDT message: "{}")'.format(self._model, exp))
        else:
            msg = ('TDT is in dummy mode. No sound or triggers will '
                   'be produced. Check TDT configuration and TDTpy '
                   'installation.')
            logger.warning(msg)  # log it
            warnings.warn(msg)  # make it red
            self.rpcox = DummyRPcoX(self._model, self._interface)

        if self.rpcox is not None:
            logger.info('Expyfun: RPcoX connection established')
        else:
            raise IOError('Problem initializing RPcoX.')
        """
        # start zBUS (may be needed for devices other than RM1)
        self.zbus = connect_zbus(interface=interface)
        if self.zbus is not None:
            logger.info('Expyfun: zBUS connection established')
        else:
            raise ExperimentError('Problem initializing zBUS.')
        """
        # load circuit
        if not self.rpcox.LoadCOF(self.circuit):
            logger.debug('Expyfun: Problem loading circuit. Clearing...')
            try:
                if self.rpcox.ClearCOF():
                    logger.debug('Expyfun: TDT circuit cleared')
                time.sleep(0.25)
                if not self.rpcox.LoadCOF(self.circuit):
                    raise RuntimeError('Second loading attempt failed')
            except:
                raise IOError('Expyfun: Problem loading circuit.')
        logger.info('Expyfun: Circuit loaded to {1} via {2}:\n{0}'
                    ''.format(self.circuit, self.model, self.interface))
        # run circuit
        if self.rpcox.Run():
            logger.info('Expyfun: TDT circuit running')
        else:
            raise SystemError('Expyfun: Problem starting TDT circuit.')
        time.sleep(0.25)
        self._set_noise_corr()
        self._set_delay(tdt_params['TDT_DELAY'],
                        tdt_params['TDT_TRIG_DELAY'])
        # Set output values to zero (esp. first few)
        for tag in ('datainleft', 'datainright'):
            self.rpcox.ZeroTag(tag)
        self.rpcox.SetTagVal('trgname', 0)
        self._used_params = tdt_params

    def _add_keyboard_init(self, ec, force_quit_keys):
        """Helper to init as keyboard"""
        # do BaseKeyboard init last, to make sure circuit is running
        Keyboard.__init__(self, ec, force_quit_keys)

# ############################### AUDIO METHODS ###############################
    def _set_noise_corr(self, val=0):
        """Helper to set the noise correlation, only -1, 0, 1 supported"""
        assert val in (-1, 0, 1)
        self.rpcox.SetTagVal('noise_corr', int(val))

    def load_buffer(self, data):
        """Load audio samples into TDT buffer.

        Parameters
        ----------
        data : np.array
            Audio data as floats scaled to (-1,+1), formatted as an Nx2 numpy
            array with dtype 'float32'.
        """
        assert data.dtype == np.float32
        # Leave the first sample zero so on reset the output goes to zero
        self.rpcox.WriteTagVEX('datainleft', 1, 'F32', data[:, 0])
        self.rpcox.WriteTagVEX('datainright', 1, 'F32', data[:, 1])
        self.rpcox.SetTagVal('nsamples', max(data.shape[0] + 1, 1))

    def play(self):
        """Send the soft trigger to start the ring buffer playback.
        """
        self.rpcox.SetTagVal('trgname', 1)
        self._trigger(1)
        logger.debug('Expyfun: Starting TDT ring buffer')

    @property
    def playing(self):
        """Is a sound currently playing"""
        return bool(int(self.rpcox.GetTagVal('playing')))

    def stop(self):
        """Send the soft trigger to stop and reset the ring buffer playback.
        """
        self._trigger(2)
        logger.debug('Expyfun: Stopping TDT audio')

    def start_noise(self):
        """Send the soft trigger to start the noise generator.
        """
        self._trigger(3)
        logger.debug('Expyfun: Starting TDT noise')

    def stop_noise(self):
        """Send the soft trigger to stop the noise generator.
        """
        self._trigger(4)
        logger.debug('Expyfun: Stopping TDT noise')

    def set_noise_level(self, new_level):
        """Set the amplitude of stationary background noise.
        """
        self.rpcox.SetTagVal('noiselev', new_level)

    def _set_delay(self, delay, delay_trig):
        """Set the delay (in ms) of the system
        """
        assert isinstance(delay, int)  # this should never happen
        assert isinstance(delay_trig, int)
        self.rpcox.SetTagVal('onsetdel', delay)
        logger.info('Expyfun: Setting TDT delay to %s' % delay)
        self.rpcox.SetTagVal('trigdel', delay_trig)
        logger.info('Expyfun: Setting TDT trigger delay to %s' % delay_trig)

# ############################### TRIGGER METHODS #############################
    def stamp_triggers(self, triggers, delay=0.03, wait_for_last=True):
        """Stamp a list of triggers with a given inter-trigger delay

        Parameters
        ----------
        triggers : list
            No input checking is done, so ensure triggers is a list,
            with each entry an integer with fewer than 8 bits (max 255).
        delay : float
            The inter-trigger delay.
        wait_for_last : bool
            If True, wait for last trigger to be stamped before returning.
        """
        for ti, trig in enumerate(triggers):
            self.rpcox.SetTagVal('trgname', trig)
            self._trigger(6)
            if ti < len(triggers) - 1 or wait_for_last:
                wait_secs(delay)

    def _trigger(self, trig):
        """Wrapper for tdt.util.RPcoX.SoftTrg()

        Parameters
        ----------
        trigger_number : int
            Trigger number to send to TDT.
        """
        if not self.rpcox.SoftTrg(trig):
            logger.warning('SoftTrg failure for trigger: {}'.format(trig))

# ############################### KEYBOARD METHODS ############################

    def _get_timebase(self):
        """Return time since circuit was started (in seconds).
        """
        return self.rpcox.GetTagVal('masterclock') / float(self.fs)

    def _clear_events(self):
        """Clear keyboard buffers.
        """
        self._trigger(7)
        self._clear_keyboard_events()

    def _retrieve_events(self, live_keys, type='presses'):
        """Values and timestamps currently in keyboard buffer.
        """
        if type != 'presses':
            raise RuntimeError("TDT Cannot get key release events")
        # get values from the tdt
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
            presses = [(v, t) for v, t in zip(press_vals, press_times)]
        else:
            presses = []
        # adds force_quit presses
        presses.extend(self._retrieve_keyboard_events([]))
        return presses

    def _correct_presses(self, events, timestamp, relative_to, kind='presses'):
        """Correct timing of presses and check for quit press"""
        events = [(k, s + self.time_correction, kind) for k, s in events]
        self.log_presses(events)
        keys = [k[0] for k in events]
        self.check_force_quit(keys)
        if timestamp:
            events = [(k, s - relative_to, t) for (k, s, t) in events]
        else:
            events = [(k, t) for (k, s, t) in events]
        return events

    def halt(self):
        """Wrapper for tdt.util.RPcoX.Halt()."""
        self.rpcox.Halt()
        logger.debug('Expyfun: Halting TDT circuit')

# ############################ READ-ONLY PROPERTIES ###########################
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
