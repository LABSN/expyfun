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
import pyglet
# don't prevent basic functionality for folks who don't use EL
try:
    import pylink
except ImportError:
    pylink = None  # analysis:ignore

from .visual import Circle, RawImage, Line, Text
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
    def __init__(self, ec=None, output_dir=None, link='default', fs=1000,
                 verbose=None):
        if pylink is None:
            raise ImportError('Could not import pylink, please ensure it '
                              'is installed correctly')
        if link == 'default':
            link = get_config('EXPYFUN_EYELINK', None)
        if fs not in [250, 500, 1000, 2000]:
            raise ValueError('fs must be 250, 500, 1000, or 2000')
        if output_dir is None:
            output_dir = os.getcwd()
        if not isinstance(output_dir, string_types):
            raise TypeError('output_dir must be a string')
        if not op.isdir(output_dir):
            os.mkdir(output_dir)
        self.output_dir = output_dir
        self._ec = ec
        logger.info('EyeLink: Initializing on {}'.format(link))
        ec.flush_logs()
        self.eyelink = pylink.EyeLink(link)
        self._file_list = []
        if self._ec is not None:
            self._size = np.array(self._ec.window_size_pix)
            self._ec._extra_cleanup_fun += [self.close]
        else:
            self._size = np.array([1920, 1200])
        self._ec.flush_logs()
        self.setup(fs)
        if 'el_id' in self._ec._id_call_dict:
            raise RuntimeError('Cannot use setup opened EL twice on EC')
        self._ec._id_call_dict['el_id'] = self._stamp_trial_id
        self._ec._ofp_critical_funs.append(self._stamp_trial_start)
        self._fake_calibration = False  # Only used for testing
        logger.debug('EyeLink: Setup complete')
        self._ec.flush_logs()

    @property
    def _is_dummy_mode(self):
        return self.eyelink.getDummyMode()

    def setup(self, fs=1000):
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
        self.command('screen_pixel_coords = ' + res_str)
        self._message('DISPLAY_COORDS ' + res_str)

        # set calibration parameters
        self.custom_calibration()

        # set parser (conservative saccade thresholds)
        self.eyelink.setSaccadeVelocityThreshold(35)
        self.eyelink.setAccelerationThreshold(9500)
        self.eyelink.setUpdateInterval(50)
        self.eyelink.setFixationUpdateAccumulate(50)
        self.command('sample_rate = {0}'.format(fs))

        # retrieve tracker version and tracker software version
        v = str(self.eyelink.getTrackerVersion())
        logger.info('Eyelink: Running experiment on a version ''{0}'' '
                    'tracker.'.format(v))
        v = LooseVersion(v).version

        # set EDF file contents
        logger.debug('EyeLink: Setting file and event filters')
        fef = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT'
        self.eyelink.setFileEventFilter(fef)
        lef = ('LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,'
               'BUTTON,FIXUPDATE,INPUT')
        self.eyelink.setLinkEventFilter(lef)
        fsf = 'LEFT,RIGHT,GAZE,HREF,AREA,GAZERES,STATUS,INPUT'
        lsf = 'LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT'
        if len(v) > 1 and v[0] == 3 and v[1] == 4:
            # remote mode possible add HTARGET ( head target)
            fsf += ',HTARGET'
            # set link data (used for gaze cursor)
            lsf += ',HTARGET'
        self.eyelink.setFileSampleFilter(fsf)
        self.eyelink.setLinkSampleFilter(lsf)

        # Ensure that we get areas
        self.eyelink.setPupilSizeDiameter('NO')

        # calibration/drift cordisp.rection target
        self.eyelink.setAcceptTargetFixationButton(5)

        # record a few samples before we actually start displaying
        # otherwise you may lose a few msec of data
        time.sleep(0.1)
        self._file_list = []

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
        file_name = datetime.datetime.now().strftime('%H%M%S')
        # make absolutely sure we don't break this
        if len(file_name) > 8:
            raise RuntimeError('filename ("{0}") is too long!\n'
                               'Must be < 8 chars'.format(file_name))
        logger.info('Eyelink: Starting recording with filename {}'
                    ''.format(file_name))
        self.eyelink.openDataFile(file_name)
        if self.eyelink.startRecording(1, 1, 1, 1) != pylink.TRIAL_OK:
            raise RuntimeError('Recording could not be started')
        #self.eyelink.waitForModeReady(1000)
        if not self.eyelink.waitForBlockStart(100, 1, 0):
            raise RuntimeError('No link samples received')
        #if not self.eyelink.isRecording():
        #    raise RuntimeError('Eyelink is not recording')
        recording = self.eyelink.getCurrentMode() == pylink.IN_RECORD_MODE
        if not self._is_dummy_mode and not recording:
            raise RuntimeError('Eyelink is not recording '
                               '{}'.format(self.eyelink.getCurrentMode()))
        self._file_list += [file_name]
        self._ec.flush_logs()
        self._toggle_dummy_cursor(True)

    def stop(self):
        """Stop Eyelink recording"""
        logger.info('Eyelink: Stopping recording')
        if self.eyelink.isConnected():
            self.eyelink.stopRecording()
            self.eyelink.closeDataFile()
        self._toggle_dummy_cursor(False)

    def calibrate(self, start='before', stop='before', beep=True):
        """Calibrate the eyetracker

        Parameters
        ----------
        start : str | None
            If ``'before'`` or ``'after'``, the recording will be started
            before or after (respectively) the calibration. If None,
            recording is not started before or afterward (manual control).
        stop : str | None
            If ``'before'`` or ``'after'``, the recording will be started
            before or after (respectively) the calibration. If None,
            recording is not started before or afterward (manual control).

        Notes
        -----
        It is recommended to use ``start = stop = 'before'`` (the default),
        as this will create new files every time a new calibration done
        (typically leading to sane file sizes) and ensure that calibrations
        are saved with the data, which is nice for post-hoc analyses.
        """
        # open file to record *before* running calibration so it gets saved!
        if start not in ['before', 'after', None]:
            raise ValueError('"start" must be "before", "after", or None, '
                             'not "{}"'.format(start))
        if stop == 'before':
            self.stop()
        if start == 'before':
            self.start()
        logger.debug('EyeLink: Entering calibration')
        self._ec.flush_logs()
        # enter Eyetracker camera setup mode, calibration and validation
        self._ec.flip()
        cal = _Calibrate(self._ec, beep)
        pylink.openGraphicsEx(cal)
        cal.setup_event_handlers()
        cal.play_beep(0)
        if not self._is_dummy_mode and not self._fake_calibration:
            self.eyelink.doTrackerSetup()
        cal.release_event_handlers()
        self._ec.flip()
        logger.debug('EyeLink: Completed calibration')
        self._ec.flush_logs()
        if stop == 'after':
            self.stop()
        if start == 'after':
            self.start()

    def _stamp_trial_id(self, ids):
        """Send trial id message

        These will be stamped as "TRIALID # # #", the suggested format.
        This should not be used for timing-critical operations; use
        ``stamp_trial_start()`` instead.

        Parameters
        ----------
        ids : list | str | int | float
            The ids to stamp. The first ID must contain at most 12
            characters when converted to a string, and the rest should
            be numbers.
        """
        # From the Pylink doc:
        #    The message should contain numbers ant text separated by spaces,
        #    with the first item containing up to 12 numbers and letters that
        #    uniquely identify the trial for analysis. Other data may follow,
        #    such as one number for each trial independent variable.
        if not isinstance(ids, list):
            ids = [ids]
        if len(str(ids[0])) > 12:
            raise ValueError('The string representation of ids[0]="{}" must '
                             'not be more than 12 characters'
                             ''.format(str(ids[0])))
        if not all([np.isscalar(x) for x in ids[1:]]):
            raise ValueError('All ids after the first must be numeric')
        ids = ' '.join([str(ii) for ii in ids])
        if len(ids) > 12:
            raise ValueError('ids must not have more than 12 characters')
        msg = 'TRIALID {}'.format(ids)
        self._message(msg)

    def _stamp_trial_start(self):
        """Signal the start of a trial

        This is a timing-critical operation used to synchronize the
        recording to stimulus presentation.
        """
        self.eyelink.sendMessage('SYNCTIME')

    def _message(self, msg):
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
            logger.info('Eyelink: saving Eyelink file: {0}\tstatus: {1}'
                        ''.format(fname, status))

    def close(self):
        """Close file and shutdown Eyelink"""
        if self.eyelink.isConnected():
            self.eyelink.stopRecording()
            self.eyelink.closeDataFile()
        self.eyelink.close()
        if 'el_id' in self._ec._id_call_dict:
            del self._ec._id_call_dict['el_id']
            idx = self._ec._ofp_critical_funs.index(self._stamp_trial_start)
            self._ec._ofp_critical_funs.pop(idx)

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
        if not fix_pos.ndim == 1 and fix_pos.size == 2:
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
                h_pix = self._size[0] * 2. / 3.
            else:
                h_pix = params['h_pix']
            if not 'v_pix' in params:
                v_pix = self._size[1] * 2. / 3.
            else:
                v_pix = params['v_pix']
            # make the locations
            mat = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])
            offsets = mat * np.array([h_pix / 2., v_pix / 2.])
            coords = (self._size / 2. + offsets)

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
        """The current eye position in pixels

        Returns
        -------
        eye_pos : array
            The current eye position. Will be [np.inf, np.inf] if the
            eye is lost.
        """
        if not self._is_dummy_mode:
            sample = self.eyelink.getNewestSample()
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
        if self._is_dummy_mode:
            self._ec.toggle_cursor(visibility)

    @property
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


if pylink is not None:
    super_class = pylink.EyeLinkCustomDisplay
else:
    super_class = object


class _Calibrate(super_class):
    """Show and control calibration screen"""
    def __init__(self, ec, beep=False):
        # set some useful parameters
        self.flush_logs = ec.flush_logs
        self.ec = ec
        self.size = np.array(ec.window_size_pix)
        self.keys = []
        self.aspect = float(self.size[0]) / self.size[1]
        self.img_span = (1.0, 1.0 * self.aspect)

        # set up reusable objects
        self.targ_circ = Circle(self.ec, radius=[7, 7],
                                units='pix', fill_color='white',
                                line_color=None)
        self.targ_ctr = Circle(self.ec, radius=[2, 2],
                               units='pix', fill_color='black',
                               line_color=None)
        self.loz_circ = Circle(self.ec, radius=[2, 2],
                               units='pix', fill_color=None,
                               line_width=2.0, line_color='white')
        self.render_disc = Circle(self.ec, radius=[2, 2],
                                  units='deg', fill_color=None,
                                  line_width=5.0, line_color='red')
        self.palette = None
        self.image_buffer = None

        # deal with parent class
        pylink.EyeLinkCustomDisplay.__init__(self)
        self.setup_cal_display = self.clear_display
        self.exit_cal_display = self.clear_display
        self.erase_cal_target = self.clear_display
        self.clear_cal_display = self.clear_display
        self.exit_image_display = self.clear_display
        self.beep = beep

    def setup_event_handlers(self):
        self.label = Text(self.ec, 'Eye Label',
                          pos=(0, -self.img_span[1] / 2.),
                          anchor_y='top', color='white')
        self.img = RawImage(self.ec, np.zeros((1, 2, 3)),
                            pos=(0, 0, self.img_span[0], self.img_span[1]),
                            units='norm')

        def on_key_press(symbol, modifiers):
            key_trans_dict = _get_key_trans_dict()
            key = key_trans_dict.get(str(symbol), symbol)
            self.keys += [pylink.KeyInput(key, modifiers)]

        # create new handler at top of handling stack
        self.ec.window.push_handlers(on_key_press=on_key_press)

    def release_event_handlers(self):
        self.ec.window.pop_handlers()  # should detacch top-level handler
        del self.label
        del self.img

    def clear_display(self):
        self.ec.flip()

    def record_abort_hide(self):
        pass

    def recenter(self, xy):
        return np.asarray(xy, float) - (self.size / 2.)

    def draw_cal_target(self, x, y):
        self.targ_circ.setPos(self.recenter((x, y)), log=True)
        self.targ_ctr.setPos(self.recenter((x, y)), log=True)
        self.targ_circ.draw()
        self.targ_ctr.draw()
        self.ec.flip()

    def render(self, x, y):
        raise NotImplementedError  # need to check this
        self.render_disc.setPos((x, y), units='px', log=False)
        self.render_disc.draw()

    def play_beep(self, eepid):
        """Play a sound during calibration/drift correct."""
        if self.beep is True:
            print('\a')

    def get_input_key(self):
        self.ec.window.dispatch_events()
        if len(self.keys) > 0:
            k = self.keys
            self.keys = []
            return k
        else:
            return None

    def get_mouse_state(self):
        return((0, 0), 0)

    def alert_printf(self, msg):
        logger.warn('EyeLink: alert_printf {}'.format(msg))

    def setup_image_display(self, w, h):
        # convert w, h from pixels to relative units
        self.img_size = np.array([w, h], float) / self.size
        self.clear_display()

    def image_title(self, text):
        self.label.setText(text, log=False)

    def set_image_palette(self, r, g, b):
        self.palette = (np.array([r, g, b], np.uint8).T).copy()

    def draw_image_line(self, width, line, totlines, buff):
        if self.image_buffer is None:
            self.image_buffer = np.empty((totlines, width, 3), np.uint8)
        self.image_buffer[line - 1, :, :] = self.palette[buff, :]
        if line == totlines:
            self.img.set_image(self.image_buffer)
            self.img.draw()
            self.label.draw()
            self.ec.flip()

    def draw_line(self, x1, y1, x2, y2, colorindex):
        # XXX check this
        print('draw_line ({0}, {1}, {2}, {3})'.format(x1, y1, x2, y2))
        color_dict = _get_color_dict()
        color = color_dict.get(str(colorindex), (0.0, 0.0, 0.0))
        x11, x22, y11, y22 = self._get_rltb(1, x1, x2, y1, y2)
        line = Line(self.ec, [x11, y11], [x22, y22],
                    lineColor=color[:3], units='pix')
        line.draw()

    def _get_rltb(self, asp, x1, x2, y1, y2):
        """Convert from image coords to screen coords"""
        r = (float)(self.half_width * 0.5 - self.half_width * 0.5 * 0.75)
        l = (float)(self.half_width * 0.5 + self.half_width * 0.5 * 0.75)
        t = (float)(self.height * 0.5 + self.height * 0.5 * asp * 0.75)
        b = (float)(self.height * 0.5 - self.height * 0.5 * asp * 0.75)
        x11 = float(float(x1) * (l - r) / float(self.img_size[0]) + r)
        x22 = float(float(x2) * (l - r) / float(self.img_size[0]) + r)
        y11 = float(float(y1) * (b - t) / float(self.img_size[1]) + t)
        y22 = float(float(y2) * (b - t) / float(self.img_size[1]) + t)
        return (x11, x22, y11, y22)

    def draw_lozenge(self, x, y, width, height, colorindex):
        # XXX check this
        print('draw lozenge ({0}, {1}, {2}, {3})'.format(x, y, width, height))
        color_dict = _get_color_dict()
        color = color_dict.get(str(colorindex), (0.0, 0.0, 0.0))
        width = int((float(width) / self.img_size[0]) * self.img_size[0])
        height = int((float(height) / self.img_size[1]) * self.img_size[1])
        r, l, t, b = self._get_rltb(1, x, x + width, y, y + height)
        xw = abs(float(l - r))
        yw = abs(float(b - t))
        rad = float(min(xw, yw) * 0.5)
        x = float(min(l, r) + rad)
        y = float(min(t, b) + rad)
        self.loz_circ.setColor(color)
        self.loz_circ.setPos(x, y, log=False)
        self.loz_circ.setRadius(rad, log=False)
        self.loz_circ.draw()


def _within_distance(pos_1, pos_2, radius):
    """Helper for checking eye position"""
    return np.sum((pos_1 - pos_2) ** 2) <= radius ** 2
