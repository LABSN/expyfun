"""Hardware interfaces for triggering"""

# Authors: Dan McCloy <drmccloy@uw.edu>
#          Eric Larson <larsoner@uw.edu>
#
# License: BSD (3-clause)

import sys
import numpy as np

from ._utils import verbose_dec, string_types, logger


class ParallelTrigger(object):
    """Parallel port and dummy triggering support.

    .. warning:: When using the parallel port, calling
                 :meth:`expyfun.ExperimentController.start_stimulus`
                 will automatically invoke a stamping of the 1 trigger, which
                 will in turn cause a delay equal to that of
                 ``trigger_duration``.
                 This can effect e.g. :class:`EyelinkController` timing.

    Parameters
    ----------
    mode : str
        'parallel' for real use. 'dummy', passes all calls.
    address : str | int | None
        The address to use. On Linux this should be a string path like
        ``'/dev/parport0'`` (equivalent to None), on Windows it should be an
        integer address like ``888`` or ``0x378`` (equivalent to None).
        The config variable ``TRIGGER_ADDRESS`` can be used to set this
        permanently.
    trigger_duration : float
        Amount of time (seconds) to leave the trigger high whenever
        sending a trigger.
    ec : instance of ExperimentController
        The ExperimentController.
    verbose : bool, str, int, or None
        If not None, override default verbose level.

    Notes
    -----
    Parallel port activation is enabled by using the ``trigger_controller``
    argument of :class:`expyfun.ExperimentController`.
    """

    @verbose_dec
    def __init__(self, mode='dummy', address=None, trigger_duration=0.01,
                 ec=None, verbose=None):
        self.ec = ec
        if mode == 'parallel':
            if sys.platform.startswith('linux'):
                address = '/dev/parport0' if address is None else address
                if not isinstance(address, string_types):
                    raise ValueError('addrss must be a string or None, got %s '
                                     'of type %s' % (address, type(address)))
                from parallel import Parallel
                logger.info('Expyfun: Using address %s' % (address,))
                self._port = Parallel(address)
                self._portname = address
                self._set_data = self._port.setData
            elif sys.platform.startswith('win'):
                from ctypes import windll
                if not hasattr(windll, 'inpout32'):
                    raise SystemError(
                        'Must have inpout32 installed, see:\n\n'
                        'http://www.highrez.co.uk/downloads/inpout32/')

                base = '0x378' if address is None else address
                logger.info('Expyfun: Using base address %s' % (base,))
                if isinstance(base, string_types):
                    base = int(base, 16)
                if not isinstance(base, int):
                    raise ValueError('address must be int or None, got %s of '
                                     'type %s' % (base, type(base)))
                self._port = windll.inpout32
                mask = np.uint8(1 << 5 | 1 << 6 | 1 << 7)
                # Use ECP to put the port into byte mode
                val = int((self._port.Inp32(base + 0x402) & ~mask) | (1 << 5))
                self._port.Out32(base + 0x402, val)

                # Now to make sure the port is in output mode we need to make
                # sure that bit 5 of the control register is not set
                val = int(self._port.Inp32(base + 2) & ~np.uint8(1 << 5))
                self._port.Out32(base + 2, val)
                self._set_data = lambda data: self._port.Out32(base, data)
                self._portname = str(base)
            else:
                raise NotImplementedError('Parallel port triggering only '
                                          'supported on Linux and Windows')
        else:  # mode == 'dummy':
            self._port = self._portname = None
            self._trigger_list = list()
            self._set_data = lambda x: (self._trigger_list.append(x)
                                        if x != 0 else None)
        self.trigger_duration = trigger_duration
        self.mode = mode

    def __repr__(self):
        return '<ParallelTrigger : %s (%s)>' % (self.mode, self._portname)

    def _stamp_trigger(self, trig):
        """Fake stamping."""
        self._set_data(int(trig))
        self.ec.wait_secs(self.trigger_duration)
        self._set_data(0)

    def stamp_triggers(self, triggers, delay=None, wait_for_last=True,
                       is_trial_id=False):
        """Stamp a list of triggers with a given inter-trigger delay.

        Parameters
        ----------
        triggers : list
            No input checking is done, so ensure triggers is a list,
            with each entry an integer with fewer than 8 bits (max 255).
        delay : float | None
            The inter-trigger-onset delay (includes "on" time).
            If None, will use twice the trigger duration (50% duty cycle).
        wait_for_last : bool
            If True, wait for last trigger to be stamped before returning.
        is_trial_id : bool
            No effect for this trigger controller.
        """
        if delay is None:
            delay = 2 * self.trigger_duration
        for ti, trig in enumerate(triggers):
            self._stamp_trigger(trig)
            if ti < len(triggers) - 1 or wait_for_last:
                self.ec.wait_secs(delay - self.trigger_duration)

    def close(self):
        """Release hardware interfaces."""
        if hasattr(self, '_port'):
            del self._port

    def __del__(self):
        return self.close()


def decimals_to_binary(decimals, n_bits):
    """Convert a sequence of decimal numbers to a sequence of binary numbers.

    Parameters
    ----------
    decimals : array-like
        Array of integers to convert. Must all be >= 0.
    n_bits : array-like
        Array of the number of bits to use to represent each decimal number.

    Returns
    -------
    binary : list
        Binary representation.

    Notes
    -----
    This function is useful for generating IDs to be stamped using the TDT.
    """
    decimals = np.array(decimals, int)
    if decimals.ndim != 1 or (decimals < 0).any():
        raise ValueError('decimals must be 1D with all nonnegative values')
    n_bits = np.array(n_bits, int)
    if decimals.shape != n_bits.shape:
        raise ValueError('n_bits must have same shape as decimals')
    if (n_bits <= 0).any():
        raise ValueError('all n_bits must be positive')
    binary = list()
    for d, b in zip(decimals, n_bits):
        if d > 2 ** b - 1:
            raise ValueError('cannot convert number {0} using {1} bits'
                             ''.format(d, b))
        binary.extend([int(bb) for bb in np.binary_repr(d, b)])
    assert len(binary) == n_bits.sum()  # make sure we didn't do something dumb
    return binary


def binary_to_decimals(binary, n_bits):
    """Convert a sequence of binary numbers to a sequence of decimal numbers.

    Parameters
    ----------
    binary : array-like
        Array of integers to convert. Must all be 0 or 1.
    n_bits : array-like
        Array of the number of bits used to represent each decimal number.

    Returns
    -------
    decimals : array-like
        Array of integers.
    """
    if not np.array_equal(binary, np.array(binary, bool)):
        raise ValueError('binary must only contain zeros and ones')
    binary = np.array(binary, bool)
    if binary.ndim != 1:
        raise ValueError('binary must be 1 dimensional')
    n_bits = np.atleast_1d(n_bits).astype(int)
    if np.any(n_bits <= 0):
        raise ValueError('n_bits must all be > 0')
    if n_bits.sum() != len(binary):
        raise ValueError('the sum of n_bits must be equal to the number of '
                         'elements in binary')
    offset = 0
    outs = []
    for nb in n_bits:
        outs.append(np.sum(binary[offset:offset + nb] *
                           (2 ** np.arange(nb - 1, -1, -1))))
        offset += nb
    assert offset == len(binary)
    return np.array(outs)
