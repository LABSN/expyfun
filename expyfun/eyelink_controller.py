"""Tools for controlling eyelink communication"""

# Authors: Eric Larson <larsoner@uw.edu>
#          Dan McCloy <drmccloy@uw.edu>
#
# License: BSD (3-clause)

import numpy as np
from os import path as op
import time
# don't prevent basic functionality for folks who don't use EL
try:
    import pylink
except ImportError:
    pylink = None
from .utils import get_config, verbose_dec
from .eyelink_calibration import _run_calibration

eye_list = ['LEFT_EYE', 'RIGHT_EYE', 'BINOCULAR']  # Used by eyeAvailable


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
        self._file_list = []
        self.el_setup()

    def setup(self, res, fs=1000):
        """Start up Eyelink

        Executes automatically on init, and needs to be run after
        el_save() if more eye tracking is desired.

        Parameters
        ----------
        res : tuple (length 2)
            The screen resolution used.
        fs : int
            The sample rate to use.
        """
        # Add comments
        self.command('add_file_preamble_text "Recorded by EyelinkController"')

        # map the gaze positions from the tracker to screen pixel positions
        res_str = '0 0 {0} {1}'.format(res[0] - 1, res[1] - 1)
        self.command('screen_pixel_coords = ' + res_str)
        self.message('DISPLAY_COORDS ' + res_str)

        # set calibration parameters
        self.set_calibration()

        # set parser (conservative saccade thresholds)
        self.command('saccade_velocity_threshold  =  35')
        self.command('saccade_acceleration_threshold  =  9500')

        if fs not in [250, 500, 1000, 2000]:
            raise ValueError('fs must be 250, 500, 1000, or 2000')
        self.command('sample_rate = {0}'.format(fs))

        # retrieve tracker version and tracker software version
        v, s = self.eyelink.getTrackerVersion()
        vsn = regexp(v, '\d', 'match')
        print 'Running experiment on a ''%s'' tracker.\n' % vs  # XXX

        # set EDF file contents
        fef = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT'
        fsd = 'LEFT,RIGHT,GAZE,HREF,AREA,GAZERES,STATUS,INPUT'
        lef = ('LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,'
               'FIXUPDATE,INPUT')
        lsd = 'LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT'
        if v == 3 and int(vsn[1]) == 4:
            # remote mode possible add HTARGET ( head target)
            fsd = fsd + ',HTARGET'
            # set link data (used for gaze cursor)
            lsd = lsd + ',HTARGET'
        self.command('file_event_filter = ' + fef)
        self.command('file_sample_data  = ' + fsd)
        self.command('link_event_filter = ' + lef)
        self.command('link_sample_data  = ' + lsd)

        # Ensure that we get areas
        self.command('pupil_size_diameter = NO')

        # calibration/drift cordisp.rection target
        self.command('button_function 5 "accept_target_fixation"')

        # record a few samples before we actually start displaying
        # otherwise you may lose a few msec of data
        time.sleep(0.1)
        self._file_list = []

    def eye_used(self):
        """Return the eye used 'left' or 'right'

        Returns
        -------
        eye : str
            'left' or 'right'.
        """
        eu = self.eyelink.eyeAvailable()
        eu = eye_list[eu] if eu >= 0 else None
        return eu

    def command(self, cmd):
        """Send Eyelink a command

        Parameters
        ----------
        cmd : str
            The command to send.

        Returns
        -------
        unknown
            The output of the command.
        """
        return self.eyelink.sendCommand(cmd)

    def start(self):
        """Start Eyelink recording"""
        self.eyelink.startRecording()
        file_name = 'EL_{0}'.format(datestr(now, 'HHMMSS'))
        if len(file_name) > 8:
            raise RuntimeError('filename ("{0}") is too long!\n'
                               'Must be < 8 chars'.format(file_name))
        self.eyelink.openDataFile(file_name)
        self._file_names += [file_name]

    def stop(self):
        """Stop Eyelink recording"""
        if self.eyelink.isConnected():
            self.eyelink.stopRecording()
            self.eyelink.closeFile()

    def calibrate(self, start=True):
        """Calibrate the eyetracker

        Parameters
        ----------
        start : bool
            If True, the recording will be started as soon as calibration
            is complete.

        Notes
        -----
        When calibrate is called, the stop() method is automatically
        executed before calibration begins.
        """
        # stop the recording
        self.stop()
        # enter Eyetracker camera setup mode, calibration and validation
        _run_calibration(self)
        # open file to record
        if start is True:
            self.start()

    def message(self, msg):
        """Send message to eyelink

        For TRIALIDs, it is suggested to use "TRIALID # # #", i.e.,
        TRIALID followed by a series of integers separated by spaces.

        Parameters
        ----------
        msg : str
            The message to stamp.
        """
        if not isinstance(msg, str):
            raise TypeError('message must be a string')
        self.message(msg)
        self.command('record_status_message "{0}"'.format(msg))

    def save(self, close=True):
        """Save data

        Parameters
        ----------
        close : bool
            If True, the close() method will be called to shut down the
            Eyelink before transferring data.
        """
        if close is True:
            self.close()
        for remote_name in self._file_list:
            fname = op.join(self.output_dir, '{1}.edf'.format(remote_name))
            status = self.eyelink.receiveDataFile(remote_name, fname))
            print ('saving Eyelink file: {0}\tstatus: {1}'
                   ''.format(fname, status))  # XXX

    def close(self):
        """Close file and shutdown Eyelink"""
        if self.eyelink.isConnected():
            self.eyelink.stopRecording()
            self.eyelink.closeDataFile()
        self.eyelink.close()

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

    def custom_calibration(self, coords):
        """Use a custom calibration sequence

        Parameters
        ----------
        coords : array
            Nx2 array of target screen coordinates ([x, y] in columns).
        """
        coords = np.asanyarray(coords)
        if not len(coords.shape) == 2 or coords.shape[1] != 2:
            raise ValueError('coords must be a 2D array with 2 columns')
        raise NotImplementedError
