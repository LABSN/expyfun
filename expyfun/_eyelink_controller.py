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
import sys
import subprocess
import time
import pyglet

# don't prevent basic functionality for folks who don't use EL
try:
    import pylink
except ImportError:
    pylink = None  # analysis:ignore

from .visual import ConcentricCircles, Circle, RawImage, Line, Text
from ._utils import get_config, verbose_dec, logger, string_types

eye_list = ['LEFT_EYE', 'RIGHT_EYE', 'BINOCULAR']  # Used by eyeAvailable


def _get_key_trans_dict():
    """Helper to translate pyglet keys to pylink codes"""
    key_trans_dict = {str(pyglet.window.key.F1): pylink.F1_KEY,
                      str(pyglet.window.key.F2): pylink.F2_KEY,
                      str(pyglet.window.key.F3): pylink.F3_KEY,
                      str(pyglet.window.key.F4): pylink.F4_KEY,
                      str(pyglet.window.key.F5): pylink.F5_KEY,
                      str(pyglet.window.key.F6): pylink.F6_KEY,
                      str(pyglet.window.key.F7): pylink.F7_KEY,
                      str(pyglet.window.key.F8): pylink.F8_KEY,
                      str(pyglet.window.key.F9): pylink.F9_KEY,
                      str(pyglet.window.key.F10): pylink.F10_KEY,
                      str(pyglet.window.key.PAGEUP): pylink.PAGE_UP,
                      str(pyglet.window.key.PAGEDOWN): pylink.PAGE_DOWN,
                      str(pyglet.window.key.UP): pylink.CURS_UP,
                      str(pyglet.window.key.DOWN): pylink.CURS_DOWN,
                      str(pyglet.window.key.LEFT): pylink.CURS_LEFT,
                      str(pyglet.window.key.RIGHT): pylink.CURS_RIGHT,
                      str(pyglet.window.key.BACKSPACE): '\b',
                      str(pyglet.window.key.RETURN): pylink.ENTER_KEY,
                      str(pyglet.window.key.ESCAPE): pylink.ESC_KEY,
                      str(pyglet.window.key.NUM_ADD): pyglet.window.key.PLUS,
                      str(pyglet.window.key.NUM_SUBTRACT):
                      pyglet.window.key.MINUS,
                      }
    return key_trans_dict


def _get_color_dict():
    """Helper to translate pylink colors to pyglet"""
    color_dict = {str(pylink.CR_HAIR_COLOR): (1.0, 1.0, 1.0),
                  str(pylink.PUPIL_HAIR_COLOR): (1.0, 1.0, 1.0),
                  str(pylink.PUPIL_BOX_COLOR): (0.0, 1.0, 0.0),
                  str(pylink.SEARCH_LIMIT_BOX_COLOR): (1.0, 0.0, 0.0),
                  str(pylink.MOUSE_CURSOR_COLOR): (1.0, 0.0, 0.0)}
    return color_dict


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
        the network location of eyelink (e.g., "100.1.1.1").
    fs : int
        Sample rate to use. Must be one of [250, 500, 1000, 2000].
    verbose : bool, str, int, or None
        If not None, override default verbose level (see expyfun.verbose).

    Returns
    -------
    el_controller : instance of EyelinkController
        The Eyelink control interface.

    Notes
    -----
    The data will be saved to the ExperimentController ``output_dir``.
    If this was `None`, data will be saved to the current working dir.
    """
    @verbose_dec
    def __init__(self, ec, link='default', fs=1000, verbose=None):
        if pylink is None:
            raise ImportError('Could not import pylink, please ensure it '
                              'is installed correctly')
        if link == 'default':
            link = get_config('EXPYFUN_EYELINK', None)
        valid_fs = (250, 500, 1000, 2000)
        if fs not in valid_fs:
            raise ValueError('fs must be one of {0}'.format(list(valid_fs)))
        output_dir = ec._output_dir
        if output_dir is None:
            output_dir = os.getcwd()
        if not isinstance(output_dir, string_types):
            raise TypeError('output_dir must be a string')
        if not op.isdir(output_dir):
            os.mkdir(output_dir)
        self._output_dir = output_dir
        self._ec = ec
        if 'el_id' in self._ec._id_call_dict:
            raise RuntimeError('Cannot use initialize EL twice')
        logger.info('EyeLink: Initializing on {}'.format(link))
        ec.flush()
        if link is not None:
            iswin = (sys.platform == 'win32')
            cmd = 'ping -n 1 -w 100' if iswin else 'fping -c 1 -t100'
            cmd = subprocess.Popen('%s %s' % (cmd, link),
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
            if cmd.returncode:
                raise RuntimeError('could not connect to Eyelink @ %s, '
                                   'is it turned on?' % link)
        self._eyelink = pylink.EyeLink(link)
        self._file_list = []
        self._size = np.array(self._ec.window_size_pix)
        self._ec._extra_cleanup_fun += [self._close]
        self._ec.flush()
        self._setup(fs)
        self._ec._id_call_dict['el_id'] = self._stamp_trial_id
        self._ec._ofp_critical_funs.append(self._stamp_trial_start)
        self._ec._on_trial_ok.append(self._stamp_trial_ok)
        self._fake_calibration = False  # Only used for testing
        self._closed = False  # to prevent double-closing
        self._current_open_file = None
        logger.debug('EyeLink: Setup complete')
        self._ec.flush()

    def _setup(self, fs=1000):
        """Start up Eyelink

        Executes automatically on init, and needs to be run after
        el_save() if further eye tracking is desired.

        Parameters
        ----------
        fs : int
            The sample rate to use.
        """
        # map the gaze positions from the tracker to screen pixel positions
        res = self._size
        res_str = '0 0 {0} {1}'.format(res[0] - 1, res[1] - 1)
        logger.debug('EyeLink: Setting display coordinates and saccade levels')
        self._command('screen_pixel_coords = ' + res_str)
        self._message('DISPLAY_COORDS ' + res_str)

        # set calibration parameters
        self.custom_calibration()

        # set parser (conservative saccade thresholds)
        self._eyelink.setSaccadeVelocityThreshold(35)
        self._eyelink.setAccelerationThreshold(9500)
        self._eyelink.setUpdateInterval(50)
        self._eyelink.setFixationUpdateAccumulate(50)
        self._command('sample_rate = {0}'.format(fs))

        # retrieve tracker version and tracker software version
        v = str(self._eyelink.getTrackerVersion())
        logger.info('Eyelink: Running experiment on a version ''{0}'' '
                    'tracker.'.format(v))
        v = LooseVersion(v).version

        # set EDF file contents
        logger.debug('EyeLink: Setting file and event filters')
        fef = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT'
        self._eyelink.setFileEventFilter(fef)
        lef = ('LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,'
               'BUTTON,FIXUPDATE,INPUT')
        self._eyelink.setLinkEventFilter(lef)
        fsf = 'LEFT,RIGHT,GAZE,HREF,AREA,GAZERES,STATUS,INPUT'
        lsf = 'LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT'
        if len(v) > 1 and v[0] == 3 and v[1] == 4:
            # remote mode possible add HTARGET ( head target)
            fsf += ',HTARGET'
            # set link data (used for gaze cursor)
            lsf += ',HTARGET'
        self._eyelink.setFileSampleFilter(fsf)
        self._eyelink.setLinkSampleFilter(lsf)

        # Ensure that we get areas
        self._eyelink.setPupilSizeDiameter('NO')

        # calibration/drift cordisp.rection target
        self._eyelink.setAcceptTargetFixationButton(5)

        # record a few samples before we actually start displaying
        # otherwise you may lose a few msec of data
        time.sleep(0.1)
        self._file_list = []
        self._fs = fs

    @property
    def dummy_mode(self):
        return self._eyelink.getDummyMode()

    @property
    def fs(self):
        return self._fs

    @property
    def _is_file_open(self):
        return (self._current_open_file is not None)

    def _open_file(self):
        """Opens a new file on the Eyelink"""
        if self._is_file_open:
            raise RuntimeError('Cannot start new file, old must be closed')
        file_name = datetime.datetime.now().strftime('%H%M%S')
        while file_name in self._file_list:
            # This should succeed in under 1 second
            file_name = datetime.datetime.now().strftime('%H%M%S')
        # make absolutely sure we don't break this, but it shouldn't ever
        # be wrong
        assert len(file_name) <= 8
        logger.info('Eyelink: Opening remote file with filename {}'
                    ''.format(file_name))
        val = self._eyelink.openDataFile(file_name)
        if val != pylink.TRIAL_OK:
            raise RuntimeError('Remote file "{0}" could not be opened: {1}'
                               ''.format(file_name, val))
        self._current_open_file = file_name
        return self._current_open_file

    def _start_recording(self):
        """Start Eyelink recording"""
        if not self._is_file_open:
            raise RuntimeError('cannot start recording without file open')
        if self._eyelink.startRecording(1, 1, 1, 1) != pylink.TRIAL_OK:
            raise RuntimeError('Recording could not be started')
        #self._eyelink.waitForModeReady(100)
        if not self._eyelink.waitForBlockStart(100, 1, 0):
            raise RuntimeError('No link samples received')
        if not self.recording:
            raise RuntimeError('Eyelink is not recording')
        # double-check
        mode = self._eyelink.getCurrentMode()
        if not self.dummy_mode and not (mode == pylink.IN_RECORD_MODE):
            raise RuntimeError('Eyelink is not recording: {0}'.format(mode))
        self._ec.flush()
        self._toggle_dummy_cursor(True)

    @property
    def recording(self):
        """Returns boolean for whether or not the Eyelink is recording"""
        return (self._eyelink.isRecording() == pylink.TRIAL_OK)

    def stop(self):
        """Stop Eyelink recording and close current file"""
        if not self.recording:
            raise RuntimeError('Cannot stop, not currently recording')
        logger.info('Eyelink: Stopping recording')
        val = self._eyelink.stopRecording()
        if val != pylink.TRIAL_OK:
            logger.warn('Recording could not be stopped: {0}'.format(val))
        logger.info('Eyelink: Closing file')
        val = self._eyelink.closeDataFile()
        if val != pylink.TRIAL_OK:
            logger.warn('File could not be closed: {0}'.format(val))
        self._current_open_file = None
        self._toggle_dummy_cursor(False)

    def calibrate(self, beep=True, prompt=True):
        """Calibrate the eyetracker

        Parameters
        ----------
        beep : bool
            If True, beep when calibration begins.
        prompt : bool
            If True, a standard screen prompt will be shown.

        Returns
        -------
        fname : str | None
            Filename on the Eyelink of the started data file.
            Will be None if start is None.

        Notes
        -----
        At the start of this function, the previous Eyelink file will be
        closed (if one is open), a new file will be opened, and recording
        will be started.
        """
        # stop recording and close old file (if open), then start new one
        if self.recording:
            self.stop()
        # open file to record *before* running calibration so it gets saved!
        fname = self._open_file()
        if prompt:
            self._ec.screen_prompt('We will now perform a screen calibration.'
                                   '<br><br>Press a button to continue.')
        fname = None
        logger.info('EyeLink: Entering calibration')
        self._ec.flush()
        # enter Eyetracker camera setup mode, calibration and validation
        self._ec.flip()
        cal = _Calibrate(self._ec, beep)
        pylink.openGraphicsEx(cal)
        cal.setup_event_handlers()
        cal.play_beep(0)
        if not (self.dummy_mode or self._fake_calibration):
            self._eyelink.doTrackerSetup()
        cal.release_event_handlers()
        self._ec.flip()
        logger.info('EyeLink: Completed calibration')
        self._ec.flush()
        self._start_recording()
        return fname

    def _stamp_trial_id(self, ids):
        """Send trial id message

        These will be stamped as "TRIALID # # #", the suggested format.
        This should not be used for timing-critical operations; use
        ``stamp_trial_start()`` instead.

        Parameters
        ----------
        ids : list of int
            The ids to stamp. Up to 12 integers may be used.
        """
        # From the Pylink doc:
        #    The message should contain numbers ant text separated by spaces,
        #    with the first item containing up to 12 numbers and letters that
        #    uniquely identify the trial for analysis. Other data may follow,
        #    such as one number for each trial independent variable.
        # Here we just force up to 12 integers for simplicity.
        if not isinstance(ids, (list, tuple)):
            raise TypeError('ids must be a list (or tuple)')
        if not all([np.isscalar(x) for x in ids]):
            raise ValueError('All ids must be numeric')
        if len(ids) > 12:
            raise ValueError('ids must not have more than 12 entries')
        ids = ' '.join([str(int(ii)) for ii in ids])
        msg = 'TRIALID {}'.format(ids)
        self._message(msg)

    def _stamp_trial_start(self):
        """Signal the start of a trial

        This is a timing-critical operation used to synchronize the
        recording to stimulus presentation.
        """
        self._eyelink.sendMessage('SYNCTIME')

    def _stamp_trial_ok(self):
        """Signal the end of a trial
        """
        self._eyelink.sendMessage('TRIAL OK')

    def _message(self, msg):
        """Send message to eyelink, must be a string"""
        self._eyelink.sendMessage(msg)
        self._command('record_status_message "{0}"'.format(msg))

    def _command(self, cmd):
        """Send Eyelink a command, must be a string"""
        return self._eyelink.sendCommand(cmd)

    def transfer_remote_file(self, remote_name):
        """Pull remote file (from Eyelink) to local machine

        Parameters
        ----------
        remote_name : str
            The filename on the Eyelink.

        Returns
        -------
        fname : str
            The filename on the local machine following the transfer.
        """
        fname = op.join(self._output_dir, '{0}.edf'.format(remote_name))
        logger.info('Eyelink: saving Eyelink file: {0} ...'
                    ''.format(remote_name))
        status = self._eyelink.receiveDataFile(remote_name, fname)
        logger.info('Eyelink: transferred {0} bytes'.format(status))
        return fname

    def _close(self):
        """Shutdown Eyelink, stopping recording & closing file if necessary"""
        fnames = list()
        if not self._closed:
            if self.recording:
                self.stop()
            # make sure files get transferred
            fnames = [self.transfer_remote_file(remote_name)
                      for remote_name in self._file_list]
            self._file_list = list()
            self._eyelink.close()
            self._closed = True
            assert 'el_id' in self._ec._id_call_dict
            del self._ec._id_call_dict['el_id']
            idx = self._ec._ofp_critical_funs.index(self._stamp_trial_start)
            self._ec._ofp_critical_funs.pop(idx)
            idx = self._ec._on_trial_ok.index(self._stamp_trial_ok)
            self._ec._on_trial_ok.pop(idx)
        return fnames

    def wait_for_fix(self, fix_pos, fix_time=0., tol=100., max_wait=np.inf,
                     check_interval=0.001, units='norm'):
        """Wait for gaze to settle within a defined region

        Parameters
        ----------
        fix_pos : tuple (length 2)
            The screen position (in pixels) required.
        fix_time : float
            Amount of time required to call a fixation.
        tol : float
            The tolerance (in pixels) to consider the target hit.
        max_wait : float
            Maximum time to wait (seconds) before returning.
        check_interval : float
            Time to use between position checks (seconds).
        units : str
            Units for `fix_pos`.

        Returns
        -------
        fix_success : bool
            Whether or not the subject successfully fixated.
        """
        # initialize eye position to be outside of target
        fix_success = False

        # sample eye position for el.fix_hold seconds
        time_in = time.time()
        time_out = time_in + max_wait
        fix_pos = np.array(fix_pos)
        if not (fix_pos.ndim == 1 and fix_pos.size == 2):
            raise ValueError('fix_pos must be a 2-element array-like vector')
        fix_pos = self._ec._convert_units(fix_pos[:, np.newaxis], units, 'pix')
        fix_pos = fix_pos[:, 0]
        while (time.time() < time_out and not
               (fix_success and time.time() - time_in >= fix_time)):
            # sample eye position
            eye_pos = self.get_eye_position()  # in pixels
            if _within_distance(eye_pos, fix_pos, tol):
                fix_success = True
            else:
                fix_success = False
                time_in = time.time()
            self._ec._response_handler.check_force_quit()
            self._ec.wait_secs(check_interval)

        return fix_success

    def custom_calibration(self, ctype='HV5', horiz=2./3., vert=2./3.,
                           units='norm'):
        """Set Eyetracker to use a custom calibration sequence

        Parameters
        ----------
        ctype : str
            Type of calibration. Currently only 'HV5' is supported.
        horiz : float
            Horizontal distance (left and right, each) to use.
        vert : float
            Vertical distance (up and down, each) to use.
        units : str
            Units to use.
        """
        allowed_types = ['HV5']
        if ctype not in allowed_types:
            raise ValueError('ctype cannot be "{0}", but must be one of {1}'
                             ''.format(ctype, allowed_types))
        horiz, vert = float(horiz), float(vert)
        xx = np.array(([0., horiz], [0., vert]))
        h_pix, v_pix = np.diff(self._ec._convert_units(xx, units, 'pix'),
                               axis=1)[:, 0]
        h_max, v_max = self._size[0] / 2., self._size[1] / 2.
        for p, m, s in zip((h_pix, v_pix), (h_max, v_max), ('horiz', 'vert')):
            if p > m:
                raise ValueError('{0} too large ({1} > {2})'
                                 ''.format(s, p, m))
        # make the locations
        mat = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])
        offsets = mat * np.array([h_pix, v_pix])
        coords = (self._size / 2. + offsets)
        n_samples = coords.shape[0]
        targs = ' '.join(['{0},{1}'.format(*c) for c in coords])
        seq = ','.join([str(x) for x in range(n_samples + 1)])
        self._command('calibration_type = {0}'.format(ctype))
        self._command('generate_default_targets = NO')
        self._command('calibration_samples = {0}'.format(n_samples))
        self._command('calibration_sequence = ' + seq)
        self._command('calibration_targets = ' + targs)
        self._command('validation_samples = {0}'.format(n_samples))
        self._command('validation_sequence = ' + seq)
        self._command('validation_targets = ' + targs)

    def get_eye_position(self):
        """The current eye position in pixels

        Returns
        -------
        eye_pos : array
            The current eye position. Will be [np.inf, np.inf] if the
            eye is lost.
        """
        if not self.dummy_mode:
            sample = self._eyelink.getNewestSample()
            if sample is None:
                raise RuntimeError('No sample data, consider starting a '
                                   'recording using el.start()')
            if sample.isBinocular():
                eye_pos = (np.array(sample.getLeftEye().getGaze()) +
                           np.array(sample.getRightEye().getGaze())) / 2.
            elif sample.isLeftSample:
                eye_pos = np.array(sample.getLeftEye().getGaze())
            elif sample.isRightSample:
                eye_pos = np.array(sample.getRightEye().getGaze())
            else:
                eye_pos = np.array([np.inf, np.inf])
            eye_pos -= (self._size / 2.)
        else:
            # use mouse, already referenced to center
            eye_pos = self._ec.get_mouse_position()
        return eye_pos

    def _toggle_dummy_cursor(self, visibility):
        """Show the cursor for dummy mode"""
        if self.dummy_mode:
            self._ec.toggle_cursor(visibility)

    @property
    def file_list(self):
        """The list of files started on the EyeLink
        """
        return self._file_list

    @property
    def eye_used(self):
        """Return the eye used 'left' or 'right'

        Returns
        -------
        eye : str
            'left' or 'right'.
        """
        eu = self._eyelink.eyeAvailable()
        eu = eye_list[eu] if eu >= 0 else None
        return eu


if pylink is not None:
    super_class = pylink.EyeLinkCustomDisplay
else:
    super_class = object


class _Calibrate(super_class):
    """Show and control calibration screen"""
    def __init__(self, ec, beep=False):
        super_class.__init__(self)
        self.__target_beep__ = None
        self.__target_beep__done__ = None
        self.__target_beep__error__ = None

        # set some useful parameters
        self.ec = ec
        self.keys = []
        ws = np.array(ec.window_size_pix)
        self.img_span = 1.5 * np.array((float(ws[0]) / ws[1], 1.))

        # set up reusable objects
        self.targ_circ = ConcentricCircles(self.ec)
        self.loz_circ = Circle(self.ec, fill_color=None, line_width=2.0)
        self.image_buffer = None

        # deal with parent class
        self.setup_cal_display = self.clear_display
        self.exit_cal_display = self.clear_display
        self.erase_cal_target = self.clear_display
        self.clear_cal_display = self.clear_display
        self.exit_image_display = self.clear_display
        self.beep = beep
        self.state = 0
        self.mouse_pos = (0, 0)
        self.img_size = (0, 0)

    def setup_event_handlers(self):
        self.label = Text(self.ec, 'Eye Label', units='norm',
                          pos=(0, -self.img_span[1] / 2.),
                          anchor_y='top', color='white')
        self.img = RawImage(self.ec, np.zeros((1, 2, 3)),
                            pos=(0, 0), units='norm')

        def on_mouse_press(x, y, button, modifiers):
            self.state = 1

        def on_mouse_motion(x, y, dx, dy):
            self.mouse_pos = (x, y)

        def on_mouse_release(x, y, button, modifiers):
            self.state = 0

        def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
            self.mouse_pos = (x, y)

        def on_key_press(symbol, modifiers):
            key_trans_dict = _get_key_trans_dict()
            key = key_trans_dict.get(str(symbol), symbol)
            self.keys += [pylink.KeyInput(key, modifiers)]

        # create new handler at top of handling stack
        self.ec.window.push_handlers(on_key_press, on_mouse_press,
                                     on_mouse_motion, on_mouse_release,
                                     on_mouse_drag)

    def release_event_handlers(self):
        self.ec.window.pop_handlers()  # should detacch top-level handler
        del self.label
        del self.img

    def clear_display(self):
        self.ec.flip()

    def record_abort_hide(self):
        pass

    def draw_cal_target(self, x, y):
        self.targ_circ.set_pos((x, y), units='pix')
        self.targ_circ.draw()
        self.ec.flip()

    def play_beep(self, eepid):
        """Play a sound during calibration/drift correct."""
        self.ec.system_beep() if self.beep else None

    def get_input_key(self):
        self.ec.window.dispatch_events()
        if len(self.keys) == 0:
            return None
        k = self.keys
        self.keys = []
        return k

    def get_mouse_state(self):
        x, y = self._win2img(self.mouse_pos[0], self.mouse_pos[1])
        return ((x, y), self.state)

    def _win2img(self, x, y):
        """Convert window coordinates to img coordinates"""
        bounds, scale = self.img.bounds, self.img.scale
        x = min(max(int((x - bounds[0]) / scale), 0), self.img_size[0])
        y = min(max(int((bounds[3] - y) / scale), 0), self.img_size[1])
        return x, y

    def _img2win(self, x, y):
        """Convert window coordinates to img coordinates"""
        bounds, scale = self.img.bounds, self.img.scale
        x = int(scale * x + bounds[0])
        y = int(bounds[3] - scale * y)
        return x, y

    def alert_printf(self, msg):
        logger.warn('EyeLink: alert_printf {}'.format(msg))

    def setup_image_display(self, w, h):
        # convert w, h from pixels to relative units
        x = np.array([[0, 0], [0, self.img_span[1]]], float)
        x = np.diff(self.ec._convert_units(x, 'norm', 'pix')[1]) / h
        self.img.set_scale(x)
        self.clear_display()

    def image_title(self, text):
        text = "<center>{0}</center>".format(text)
        self.label = Text(self.ec, text, units='norm', anchor_y='top',
                          color='white', pos=(0, -self.img_span[1] / 2.))

    def set_image_palette(self, r, g, b):
        self.palette = np.array([r, g, b], np.uint8).T

    def draw_image_line(self, width, line, totlines, buff):
        if self.image_buffer is None:
            self.img_size = (width, totlines)
            self.image_buffer = np.empty((totlines, width, 3), float)
        self.image_buffer[line - 1, :, :] = self.palette[buff, :] / 255.
        if line == totlines:
            self.img.set_image(self.image_buffer)
            self.img.draw()
            self.label.draw()
            self.draw_cross_hair()
            self.ec.flip()

    def draw_line(self, x1, y1, x2, y2, colorindex):
        color = _get_color_dict()[str(colorindex)]
        x1, y1 = self._img2win(x1, y1)
        x2, y2 = self._img2win(x2, y2)
        Line(self.ec, ((x1, x2), (y1, y2)), 'pix', color).draw()

    def draw_lozenge(self, x, y, width, height, colorindex):
        coords = self._img2win(x + width / 2., y + width / 2.)
        width = width * self.img.scale / 2.
        height = height * self.img.scale / 2.
        self.loz_circ.set_line_color(_get_color_dict()[str(colorindex)])
        self.loz_circ.set_pos(coords, units='pix')
        self.loz_circ.set_radius((width, height), units='pix')
        self.loz_circ.draw()


def _within_distance(pos_1, pos_2, radius):
    """Helper for checking eye position"""
    return np.sum((pos_1 - pos_2) ** 2) <= radius ** 2
