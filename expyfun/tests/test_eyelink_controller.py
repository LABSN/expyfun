from nose.tools import assert_raises, assert_true
import warnings

from expyfun import EyelinkController, ExperimentController
from expyfun._utils import _TempDir, requires_pylink, _hide_window

warnings.simplefilter('always')

std_args = ['test']
temp_dir = _TempDir()
std_kwargs = dict(output_dir=temp_dir, full_screen=False, window_size=(1, 1),
                  participant='foo', session='01', noise_db=0)


@_hide_window
@requires_pylink
def test_eyelink_methods():
    """Test EL methods
    """
    ec = ExperimentController(*std_args, **std_kwargs)
    assert_raises(ValueError, EyelinkController, ec, fs=999)
    el = EyelinkController(ec)
    assert_raises(RuntimeError, EyelinkController, ec)  # can't have two open
    assert_raises(ValueError, el.custom_calibration, ctype='hey')
    el.custom_calibration()
    el._open_file()
    el._start_recording()
    el.get_eye_position()
    assert_raises(ValueError, el.wait_for_fix, [1])
    x = el.wait_for_fix([-10000, -10000], max_wait=0.1)
    assert_true(x is False)
    assert el.eye_used
    print(el.file_list)
    # run much of the calibration code, but don't *actually* do it
    el._fake_calibration = True
    el.calibrate(beep=False, prompt=False)
    el._fake_calibration = False
    # missing el_id
    assert_raises(KeyError, ec.identify_trial, ec_id='foo', ttl_id=[0])
    ec.identify_trial(ec_id='foo', ttl_id=[0], el_id=[1])
    ec.flip_and_play()
    ec.identify_trial(ec_id='foo', ttl_id=[0], el_id=[1, 1])
    ec.flip_and_play()
    assert_raises(ValueError, ec.identify_trial, ec_id='foo', ttl_id=[0],
                  el_id=[1, dict()])
    assert_raises(ValueError, ec.identify_trial, ec_id='foo', ttl_id=[0],
                  el_id=[0] * 13)
    assert_raises(TypeError, el._message, 1)
    el.stop()
    assert_true(not el._closed)
    ec.close()  # auto-calls el.close()
    assert_true(el._closed)
