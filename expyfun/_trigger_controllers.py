"""Hardware interfaces for triggering"""

# Authors: Dan McCloy <drmccloy@uw.edu>
#          Eric Larson <larsoner@uw.edu>
#
# License: BSD (3-clause)

import numpy as np
import platform

from ._utils import wait_secs, verbose_dec


class ParallelTrigger(object):
    """Parallel port and dummy triggering support

    IMPORTANT: When using the parallel port, note that calling
    ec.flip_and_play() will automatically invoke a stamping of
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

    On Windows, you may need to download ``inpout32.dll`` from someplace
    like:

        http://logix4u.net/InpOutBinaries.zip
    """
    @verbose_dec
    def __init__(self, mode='dummy', address=None, high_duration=0.001,
                 verbose=None):
        self.parallel = None
        if mode == 'parallel':
            self._stamp_trigger = self._parallel_trigger
            if 'Linux' in platform.system():
                address = '/dev/parport0' if address is None else address
                try:
                    import parallel as _p
                except ImportError:
                    raise ImportError('must have module "parallel" installed '
                                      'to use parallel triggering on Linux')
                else:
                    self._port = _p.Parallel(address)
                    self._set_data = self._port.setData
            elif 'Windows' in platform.system():
                from ctypes import windll
                if not hasattr(windll, 'inpout32'):
                    raise SystemError('Must have inpout32 installed')

                addr = 0x0378 if address is None else address
                if isinstance(addr, basestring) and addr.startswith('0x'):
                    base = int(addr, 16)
                else:
                    base = addr

                self._port = windll.inpout32
                mask = np.uint8(1 << 5 | 1 << 6 | 1 << 7)
                # Use ECP to put the port into byte mode
                val = int((self._port.Inp32(base + 0x402) & ~mask) | (1 << 5))
                self.port.Out32(base + 0x402, val)

                # Now to make sure the port is in output mode we need to make
                # sure that bit 5 of the control register is not set
                val = int(self._port.Inp32(base + 2) & ~np.uint8(1 << 5))
                self._port.Out32(base + 2, val)

                def _set_data(data):
                    return self._port.Out32(base, data)
                self._set_data = _set_data
            else:
                raise NotImplementedError
        else:  # mode == 'dummy':
            self._stamp_trigger = self._dummy_trigger

        self.high_duration = high_duration

    def _dummy_trigger(self, trig):
        """Fake stamping"""
        pass

    def _parallel_trigger(self, trig):
        """Stamp a single byte via parallel port"""
        self._set_data(int(trig))
        wait_secs(self.high_duration)
        self._set_data(0)

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
            self._stamp_trigger(trig)
            if ti < len(triggers) - 1:
                wait_secs(delay - self.high_duration)

    def close(self):
        """Release hardware interfaces
        """
        if self.parallel is not None:
            del self.parallel

    def __del__(self):
        """Nice cleanup"""
        if hasattr(self, '_port'):
            del self._port
