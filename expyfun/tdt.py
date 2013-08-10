import time
import numpy as np
import platform
if platform.platform == 'Windows':
    from tdt.util import connect_rpcox, connect_zbus
else:
    connect_rpcox = None
    connect_zbus = None
from psychopy import logging as psylog
from .utils import get_config


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
            self.rpcox = connect_rpcox(name=self.tdt_type,
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
                        ''.format(self.circuit, self.tdt_type, self.interface))
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
        When using PsychoPy for audio, fs is potentially user-settable, but
        defaults to 22050 Hz.  When using TDT for audio, fs is read from the
        TDT circuit.
        """
        return np.float(self.rpcox.GetSFreq())

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

    def write_buffer(self, data, buffer_name, offset=0):
        """Wrapper for tdt.util.RPcoX.WriteTagV()
        """
        # TODO: check to make sure data is properly formatted / correct dtype
        # check dimensions of array
        # cast as np.float32 with order='C'
        self.rpcox.WriteTagV(buffer_name, offset, data)

    def clear_buffer(self, buffer_name):
        """Wrapper for tdt.util.RPcoX.ZeroTag()
        """
        self.rpcox.ZeroTag(buffer_name)

    def halt_circuit(self):
        """Wrapper for tdt.util.RPcoX.Halt()
        """
        self.rpcox.Halt()


def get_tdt_rates():
    return {'6k': 6103.515625, '12k': 12207.03125, '25k': 24414.0625,
            '50k': 48828.125, '100k': 97656.25, '200k': 195312.5}
