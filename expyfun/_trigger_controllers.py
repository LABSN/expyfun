"""Hardware interfaces for triggering"""

# Authors: Dan McCloy <drmccloy@uw.edu>
#          Eric Larson <larsoner@uw.edu>
#
# License: BSD (3-clause)

import numpy as np

from ._utils import wait_secs, verbose_dec


class ParallelTrigger(object):
    """Parallel port and dummy triggering support

    IMPORTANT: When using the parallel port, note that calling
    ec.start_stimulus() will automatically invoke a stamping of
    the 1 trigger, which will cause a delay equal to that of
    high_duration.

    Parameters
    ----------
    mode : str
        'parallel' for real use. 'dummy', passes all calls.
    address : str | None
        The address to use. On Linux this should be a path like
        '/dev/parport0', on Windows it should be an address like
        888 (a.k.a. 0x0378).
    high_duration : float
        Amount of time (seconds) to leave the trigger high whenever
        sending a trigger.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see expyfun.verbose).

    Notes
    -----
    On Linux, parallel port may require some combination of the following:

        1. ``sudo modprobe ppdev``
        2. Add user to ``lp`` group (``/etc/group``)
        3. Run ``sudo rmmod lp`` (otherwise ``lp`` takes exclusive control)
        4. Edit ``/etc/modprobe.d/blacklist.conf`` to add ``blacklist lp``

    The ``parallel`` module must also be installed.

    On Windows, you may need to download ``inpout32.dll`` from someplace
    like:

        http://logix4u.net/InpOutBinaries.zip
    """
    @verbose_dec
    def __init__(self, mode='dummy', address=None, high_duration=0.001,
                 verbose=None):
        if mode == 'parallel':
            raise NotImplementedError('Parallel port triggering has not '
                                      'been sufficiently tested')
            #self._stamp_trigger = self._parallel_trigger
            #if 'Linux' in platform.system():
            #    address = '/dev/parport0' if address is None else address
            #    import parallel as _p
            #    self._port = _p.Parallel(address)
            #    self._set_data = self._port.setData
            #elif 'Windows' in platform.system():
            #    from ctypes import windll
            #    if not hasattr(windll, 'inpout32'):
            #        raise SystemError('Must have inpout32 installed')

            #    addr = 0x0378 if address is None else address
            #    base = int(addr, 16) if addr[:2] == '0x' else addr
            #    self._port = windll.inpout32
            #    mask = np.uint8(1 << 5 | 1 << 6 | 1 << 7)
            #    # Use ECP to put the port into byte mode
            #    val = int((self._port.Inp32(base + 0x402) & ~mask) | (1 << 5))
            #    self.port.Out32(base + 0x402, val)

            #    # Now to make sure the port is in output mode we need to make
            #    # sure that bit 5 of the control register is not set
            #    val = int(self._port.Inp32(base + 2) & ~np.uint8(1 << 5))
            #    self._port.Out32(base + 2, val)

            #    def _set_data(data):
            #        return self._port.Out32(base, data)
            #    self._set_data = _set_data
            #else:
            #    raise NotImplementedError
        else:  # mode == 'dummy':
            self._stamp_trigger = self._dummy_trigger
        self.high_duration = high_duration

    def _dummy_trigger(self, trig):
        """Fake stamping"""
        pass

    #def _parallel_trigger(self, trig):
    #    """Stamp a single byte via parallel port"""
    #    self._set_data(int(trig))
    #    wait_secs(self.high_duration)
    #    self._set_data(0)

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
            self._stamp_trigger(trig)
            if ti < len(triggers) - 1 or wait_for_last:
                wait_secs(delay - self.high_duration)

    def close(self):
        """Release hardware interfaces
        """
        if hasattr(self, '_port'):
            del self._port


def decimals_to_binary(decimals, n_bits):
    """Convert a sequence of decimal numbers to a sequence of binary numbers

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
    """Convert a sequence of binary numbers to a sequence of decimal numbers

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
