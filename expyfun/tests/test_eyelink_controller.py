from nose.tools import assert_raises, assert_true
import warnings

from expyfun import EyelinkController, ExperimentController
from expyfun._utils import _TempDir, _hide_window, requires_opengl21

warnings.simplefilter('always')

std_args = ['test']
temp_dir = _TempDir()
std_kwargs = dict(output_dir=temp_dir, full_screen=False, window_size=(1, 1),
                  participant='foo', session='01', noise_db=0)


@_hide_window
@requires_opengl21
def test_eyelink_methods():
    """Test EL methods
    """
    with ExperimentController(*std_args, **std_kwargs) as ec:
        assert_raises(ValueError, EyelinkController, ec, fs=999)
        el = EyelinkController(ec)
        assert_raises(RuntimeError, EyelinkController, ec)  # can't have 2 open
        assert_raises(ValueError, el.custom_calibration, ctype='hey')
        el.custom_calibration()
        el._open_file()
        assert_raises(RuntimeError, el._open_file)
        el._start_recording()
        el.get_eye_position()
        assert_raises(ValueError, el.wait_for_fix, [1])
        x = el.wait_for_fix([-10000, -10000], max_wait=0.1)
        assert_true(x is False)
        assert el.eye_used
        print(el.file_list)
        assert_true(len(el.file_list) > 0)
        print(el.fs)
        # run much of the calibration code, but don't *actually* do it
        el._fake_calibration = True
        el.calibrate(beep=False, prompt=False)
        el._fake_calibration = False
        # missing el_id
        assert_raises(KeyError, ec.identify_trial, ec_id='foo', ttl_id=[0])
        ec.identify_trial(ec_id='foo', ttl_id=[0], el_id=[1])
        ec.start_stimulus()
        ec.stop()
        ec.trial_ok()
        ec.identify_trial(ec_id='foo', ttl_id=[0], el_id=[1, 1])
        ec.start_stimulus()
        ec.stop()
        ec.trial_ok()
        assert_raises(ValueError, ec.identify_trial, ec_id='foo', ttl_id=[0],
                      el_id=[1, dict()])
        assert_raises(ValueError, ec.identify_trial, ec_id='foo', ttl_id=[0],
                      el_id=[0] * 13)
        assert_raises(TypeError, ec.identify_trial, ec_id='foo', ttl_id=[0],
                      el_id=dict())
        assert_raises(TypeError, el._message, 1)
        el.stop()
        el.transfer_remote_file(el.file_list[0])
        assert_true(not el._closed)
    # ec.close() auto-calls el.close()
    assert_true(el._closed)
