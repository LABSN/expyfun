"""Hardware interfaces for triggering"""

# Authors: Dan McCloy <drmccloy@uw.edu>
#          Eric Larson <larsoner@uw.edu>
#
# License: BSD (3-clause)

import platform

from ._utils import wait_secs, verbose_dec, psylog


class PsychTrigger(object):
    """Parallel port and dummy triggering support

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
    def __init__(self, mode='dummy', address=None, high_duration=0.01,
                 verbose=None):
        self.parallel = None
        if mode == 'parallel':
            # use nested import in case parallel isn't used
            from psychopy import parallel
            self._stamp_trigger = self._parallel_trigger
            # Psychopy has some legacy methods (e.g., parallel.setData()),
            # but we triage here to save time when time-critical stamping
            # may be used
            if 'Linux' in platform.system():
                address = '/dev/parport0' if address is None else address
                try:
                    import parallel as _p
                    assert _p
                except ImportError:
                    raise ImportError('must have module "parallel" installed '
                                      'to use parallel triggering on Linux')
                else:
                    self.parallel = parallel.PParallelLinux(address)
            elif 'Windows' in platform.system():
                address = 0x0378 if address is None else float(address)
                self.parallel = parallel.PParallelInpOut32(address)
            else:
                raise NotImplementedError
        else:  # mode == 'dummy':
            self._stamp_trigger = self._dummy_trigger

        self.high_duration = high_duration

    def _dummy_trigger(self, trig):
        """Fake stamping"""
        wait_secs(self.high_duration)

    def _parallel_trigger(self, trig):
        """Stamp a single byte via parallel port"""
        self.parallel.setData(int(trig))
        wait_secs(self.high_duration)
        self.parallel.setData(0)

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
            if ti < len(triggers):
                self._stamp_trigger(trig)
                wait_secs(delay - self.high_duration)

    def close(self):
        """Release hardware interfaces
        """
        if self.parallel is not None:
            del self.parallel
