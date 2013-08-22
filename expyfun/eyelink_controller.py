"""Tools for controlling eyelink communication"""

# Authors: Eric Larson <larsoner@uw.edu>
#          Dan McCloy <drmccloy@uw.edu>
#
# License: BSD (3-clause)

import numpy as np
import datetime
from distutils.version import LooseVersion
import os
from os import path as op
import time
from psychopy import logging as psylog
# don't prevent basic functionality for folks who don't use EL
try:
    import pylink
except ImportError:
    pylink = None
from .utils import get_config, verbose_dec
from .eyelink_core_graphics_pyglet import CalibratePyglet
eye_list = ['LEFT_EYE', 'RIGHT_EYE', 'BINOCULAR']  # Used by eyeAvailable


class EyelinkController(object):
    """Eyelink communication and control methods

    Parameters
    ----------
    ec : instance of ExperimentController | None
        ExperimentController instance to interface with. Necessary for
        doing calibrations.
    link : str | None
        If 'default', the default value will be read from EXPYFUN_EYELINK.
        If None, dummy (simulation) mode will be used. If str, should be
        the network location of eyelink (e.g., "100.0.0.1").
    output_dir : str | None
        Directory to store the output files in. If None, will use CWD.
    fs : int
        Sample rate to use. Must be one of [250, 500, 1000, 2000].
    verbose : bool, str, int, or None
        If not None, override default verbose level (see expyfun.verbose).

    Returns
    -------
    el_controller : instance of EyelinkController
        The Eyelink control interface.
    """
    @verbose_dec
    def __init__(self, ec=None, output_dir=None, link='default', fs=1000):
        if pylink is None:
            raise ImportError('Could not import pylink, please ensure it '
                              'is installed correctly')
        if link == 'default':
            link = get_config('EXPYFUN_EYELINK', None)
        if output_dir is None:
            output_dir = os.getcwd()
        if not isinstance(output_dir, basestring):
            raise TypeError('output_dir must be a string')
        if not op.isdir(output_dir):
            os.mkdir(output_dir)
        self.output_dir = output_dir
        self._ec = ec
        self.eyelink = pylink.EyeLink(link)
        self._file_list = []
        self._display_res = self._ec._win.size.copy()
        self.setup(fs)

    @property
    def _is_dummy_mode(self):
        return self.eyelink.getDummyMode()

    def setup(self, fs=1000):
        """Start up Eyelink

        Executes automatically on init, and needs to be run after
        el_save() if more eye tracking is desired.

        Parameters
        ----------
        fs : int
            The sample rate to use.
        """
        # Add comments
        self.command('add_file_preamble_text "Recorded by EyelinkController"')

        # map the gaze positions from the tracker to screen pixel positions
        res = self._display_res
        res_str = '0 0 {0} {1}'.format(res[0] - 1, res[1] - 1)
        self.command('screen_pixel_coords = ' + res_str)
        self.message('DISPLAY_COORDS ' + res_str)

        # set calibration parameters
        self.custom_calibration()

        # set parser (conservative saccade thresholds)
        self.command('saccade_velocity_threshold  =  35')
        self.command('saccade_acceleration_threshold  =  9500')

        if fs not in [250, 500, 1000, 2000]:
            raise ValueError('fs must be 250, 500, 1000, or 2000')
        self.command('sample_rate = {0}'.format(fs))

        # retrieve tracker version and tracker software version
        v = str(self.eyelink.getTrackerVersion())
        psylog.info('Running experiment on a ''{0}'' tracker.'.format(v))
        v = LooseVersion(v).version

        # set EDF file contents
        fef = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT'
        fsd = 'LEFT,RIGHT,GAZE,HREF,AREA,GAZERES,STATUS,INPUT'
        lef = ('LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,'
               'FIXUPDATE,INPUT')
        lsd = 'LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT'
        if len(v) > 1 and v[0] == 3 and v[1] == 4:
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
        """Start Eyelink recording

        Notes
        -----
        Filenames are saved by HHMMSS format, DO NOT start and stop
        recordings anywhere near once per second.
        """
        self.eyelink.startRecording(1, 1, 1, 1)
        file_name = datetime.datetime.now().strftime('%H%M%S')
        # make absolutely sure we don't break this
        if len(file_name) > 8:
            raise RuntimeError('filename ("{0}") is too long!\n'
                               'Must be < 8 chars'.format(file_name))
        self.eyelink.openDataFile(file_name)
        self._file_list += [file_name]

    def stop(self):
        """Stop Eyelink recording"""
        if self.eyelink.isConnected():
            self.eyelink.stopRecording()
            self.eyelink.closeDataFile()

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
        #if not self._is_dummy_mode:
        self._ec.clear_screen()
        genv = CalibratePyglet(self._ec.win.winHandle)
        pylink.openGraphicsEx(genv)
        genv.setup_event_handlers()
        self.eyelink.doTrackerSetup()
        genv.release_event_handlers()
        self._ec.clear_screen()
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
        self.eyelink.sendMessage(msg)
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
            fname = op.join(self.output_dir, '{0}.edf'.format(remote_name))
            status = self.eyelink.receiveDataFile(remote_name, fname)
            psylog.info('saving Eyelink file: {0}\tstatus: {1}'
                        ''.format(fname, status))

    def close(self):
        """Close file and shutdown Eyelink"""
        if self.eyelink.isConnected():
            self.eyelink.stopRecording()
            self.eyelink.closeDataFile()
        self.eyelink.close()

    def wait_for_fix(self, fix_pos, fix_time=1e-4, tol=15, max_wait=np.inf):
        """Wait for gaze to settle within a defined region

        Parameters
        ----------
        fix_pos : tuple (length 2)
            The screen position (in pixels) required.
        fix_time : float
            Amount of time required to call a fixation.
        tol : int
            The tolerance (in pixels) to consider the target hit.
        max_wait : float
            Maximum time to wait (seconds) before returning.

        Returns
        -------
        fix_success : bool
            Whether or not the subject successfully fixated.
        """
        self._toggle_dummy_cursor(True)
        # initialize eye position to be outside of target
        fix_success = False

        # sample eye position for el.fix_hold seconds
        time_out = time.time() + max_wait
        while time.time() < time_out and not fix_success:
            # sample eye position
            eye_pos = self.get_eye_position()
            if _within_distance(eye_pos, fix_pos, tol):
                fix_success = True

        self._toggle_dummy_cursor(False)
        return fix_success

    def hold_Fix(self, fix_pos, hold_duration, tol=15):
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
        # sample eye position for el.fix_hold seconds
        self._toggle_dummy_cursor(True)
        stop_time = time.time() + hold_duration
        fix_success = True
        while time.time() < stop_time:
            # sample eye position
            eye_pos = self.get_eye_position()
            if not _within_distance(eye_pos, fix_pos, tol):
                fix_success = False
        self._toggle_dummy_cursor(False)
        return fix_success

    def custom_calibration(self, params=None):
        """Set Eyetracker to use a custom calibration sequence

        Parameters
        ----------
        params : dict | None
            Type of calibration to use. Must have entries 'type' (must be
            'HV5') and h_pix, v_pix for total span in both directions. If
            h_pix and v_pix are not defined, 2/3 of the screen will be used.
            If params is None, a simple HV5 calibration will be used.
        """
        if params is None:
            params = dict(type='HV5')
        if not isinstance(params, dict):
            raise TypeError('parameters must be a dict')
        if 'type' not in params:
            raise KeyError('"type" must be an entry in parameters')
        allowed_types = ['HV5']
        if not params['type'] in allowed_types:
            raise ValueError('params["type"] cannot be "{0}", but must be '
                             ' one of {1}'.format(params['type'],
                                                  allowed_types))

        if params['type'] == 'HV5':
            if not 'h_pix' in params:
                h_pix = self._display_res[0] * 2. / 3.
            else:
                h_pix = params['h_pix']
            if not 'v_pix' in params:
                v_pix = self._display_res[1] * 2. / 3.
            else:
                v_pix = params['v_pix']
            # make the locations
            mat = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])
            offsets = mat * np.array([h_pix / 2., v_pix / 2.])
            coords = (self._display_res / 2. + offsets)

        n_samples = coords.shape[0]
        targs = ' '.join(['{0},{1}'.format(*c) for c in coords])
        seq = ','.join([str(x) for x in range(n_samples + 1)])
        self.command('calibration_type = {0}'.format(params['type']))
        self.command('generate_default_targets = NO')
        self.command('calibration_samples = {0}'.format(n_samples))
        self.command('calibration_sequence = ' + seq)
        self.command('calibration_targets = ' + targs)
        self.command('validation_samples = {0}'.format(n_samples))
        self.command('validation_sequence = ' + seq)
        self.command('validation_targets = ' + targs)

    def get_eye_position(self):
        if not self._is_dummy_mode:
            sample = self.eyelink.getNewestSample()
            left_eye_pos = sample.getLeftEye()
            right_eye_pos = sample.getRightEye()
            if all(left_eye_pos != -32768) and all(right_eye_pos != -32768):
                eye_pos = [-32768, -32768]
            elif self.eye_used == 'LEFT_EYE':
                if all(left_eye_pos == -32768):
                    # use right eye instead
                    eye_pos = right_eye_pos
                else:
                    eye_pos = left_eye_pos
            elif self.eye_used == 'RIGHT_EYE':
                if all(right_eye_pos == -32768):
                    # use left eye instead
                    eye_pos = left_eye_pos
                else:
                    eye_pos = right_eye_pos
            else:
                eye_pos = -32768
        else:
            # use mouse, referenced to lower left
            eye_pos = self._ec.get_mouse_position() + (self._display_res / 2.)
        return eye_pos

    def _toggle_dummy_cursor(self, visibility):
        """Show the cursor for dummy mode"""
        if self._is_dummy_mode:
            self._ec.toggle_cursor(visibility)


def _within_distance(pos_1, pos_2, radius):
    return np.sum((pos_1 - pos_2) ** 2) <= radius ** 2
