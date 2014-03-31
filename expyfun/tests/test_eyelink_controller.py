from nose.tools import assert_raises, assert_true
import warnings
from os import path as op

from expyfun import EyelinkController, ExperimentController
from expyfun._utils import _TempDir, requires_pylink

warnings.simplefilter('always')

std_args = ['test']
temp_dir = _TempDir()
std_kwargs = dict(output_dir=temp_dir, full_screen=False, window_size=(1, 1),
                  participant='foo', session='01', noise_db=0)


@requires_pylink
def test_eyelink_methods():
    """Test EL methods
    """
    with ExperimentController(*std_args, **std_kwargs) as ec:
        assert_raises(TypeError, EyelinkController, ec, output_dir=1)
        assert_raises(ValueError, EyelinkController, ec, fs=999,
                      output_dir=temp_dir)
        el = EyelinkController(ec, output_dir=op.join(temp_dir, 'test'))
        assert_raises(RuntimeError, EyelinkController, ec)  # can't have 2 open
        assert_raises(TypeError, el.custom_calibration, 'blah')
        assert_raises(KeyError, el.custom_calibration, dict(me='hey'))
        assert_raises(ValueError, el.custom_calibration, dict(type='hey'))
        el.custom_calibration(dict(type='HV5', h_pix=10, v_pix=10))
        el.get_eye_position()
        assert_raises(ValueError, el.wait_for_fix, [1])
        x = el.wait_for_fix([-10000, -10000], max_wait=0.1)
        assert_true(x is False)
        assert el.eye_used
        # run much of the calibration code, but don't *actually* do it
        el._fake_calibration = True
        assert_raises(ValueError, el.calibrate, start='foo', beep=False)
        assert_raises(ValueError, el.calibrate, stop='foo', beep=False)
        el.calibrate(beep=False)
        el.calibrate(start='after', stop='after', beep=False)
        el._fake_calibration = False
        # missing el_id
        assert_raises(KeyError, ec.identify_trial, ec_id='foo', ttl_id=[0])
        ec.identify_trial(ec_id='foo', ttl_id=[0], el_id=1)
        ec.flip_and_play()
        ec.identify_trial(ec_id='foo', ttl_id=[0], el_id=[1, 1])
        ec.flip_and_play()
        assert_raises(ValueError, ec.identify_trial, ec_id='foo', ttl_id=[0],
                      el_id=[1, dict()])
        assert_raises(ValueError, ec.identify_trial, ec_id='foo', ttl_id=[0],
                      el_id=[0] * 13)
        assert_raises(TypeError, el._message, 1)
        el.stop()
        el.save()
    # ec.close() auto-calls el.close()
