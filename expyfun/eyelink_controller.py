"""Tools for controlling eyelink communication"""

# Authors: Eric Larson <larsoner@uw.edu>
#          Dan McCloy <drmccloy@uw.edu>
#
# License: BSD (3-clause)

# don't prevent basic functionality for folks who don't use EL
try:
    import pylink
except ImportError:
    pylink = None
from .utils import get_config, verbose_dec


class EyelinkController(object):
    """Eyelink communication and control methods

    Parameters
    ----------
    link : str | None
        If 'default', the default value will be read from EXPYFUN_EYELINK.
        If None, dummy (simulation) mode will be used. If str, should be
        the network location of eyelink (e.g., "100.0.0.1").
    verbose : bool, str, int, or None
        If not None, override default verbose level (see expyfun.verbose).

    Returns
    -------
    exp_controller : instance of ExperimentController
        The experiment control interface.
    """
    @verbose_dec
    def __init__(self, link='default'):
        if pylink is None:
            raise ImportError('Could not import pylink, please ensure it '
                              'is installed correctly')
        if link == 'default':
            link = get_config('EXPYFUN_EYELINK', None)
        self.eyelink = pylink.EyeLink(link)
        self.el_setup()

    def setup(self):
        """Start up Eyelink

        Executes automatically on init, and needs to be run after
        el_save() if more eye tracking is desired.
        """
        raise NotImplementedError

    def start(self):
        """Start Eyelink recording"""
        self.eyelink.startRecording()

    def stop(self):
        """Stop Eyelink recording"""
        self.eyelink.stopRecording()

    def calibrate(self):
        """Calibrate the eyetracker
        """
        raise NotImplementedError

    def send_message(self, message):
        """Send message to eyelink

        For TRIALIDs, it is suggested to use "TRIALID # # #", i.e.,
        TRIALID followed by a series of integers separated by spaces.

        Parameters
        ----------
        message : str
            The message to stamp.
        """
        if not isinstance(message, str):
            raise TypeError('message must be a string')
        self.eyelink.message(message)
        self.eyelink.command('record_status_message "{0}"'.format(message))

    def save(self):
        """Save data and shutdown Eyelink"""

    def close(self):
        """Close file and shutdown Eyelink"""
        if self.eyelink.isConnected():
            self.eyelink.stopRecording()
            self.eyelink.closeFile()
        self.eyelink.shutdown()

    def wait_for_fix(self, fix_pos, tol=15):
        """Wait for gaze to settle within a defined region

        Parameters
        ----------
        fix_pos : tuple (length 2)
            The screen position (in pixels) required.
        tol : int
            The tolerance (in pixels) to consider the target hit.

        Returns
        -------
        successful : bool
            Whether or not the subject successfully fixated.
        """
        raise NotImplementedError

    def hold_Fix(el, fix_pos, hold_duration, tol=15):
        """Require the user to maintain gaze

        Parameters
        ----------
        fix_pos : tuple (length 2)
            The screen position (in pixels) required.
        hold_duration : float
            Duration the user must hold their position.
        tol : int
            The tolerance (in pixels) to consider the target hit.

        Returns
        -------
        successful : bool
            Whether or not the subject successfully fixated.
        """
        raise NotImplementedError

    def custom_calibration(coordinates):
        """Use a custom calibration sequence

        Parameters
        ----------
        coordinates : array
            Nx2 array of target screen coordinates ([x, y] in columns).
        """
        raise NotImplementedError
