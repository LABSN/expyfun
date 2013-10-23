from nose.tools import assert_raises, assert_true
import warnings

from expyfun import EyelinkController, ExperimentController
from expyfun._utils import _TempDir, requires_pylink

warnings.simplefilter('always')

std_args = ['test']
temp_dir = _TempDir()
std_kwargs = dict(output_dir=temp_dir, full_screen=False, window_size=(1, 1),
                  participant='foo', session='01')


@requires_pylink
def test_eyelink_methods():
    """Test EL methods
    """
    ec = ExperimentController(*std_args, **std_kwargs)
    assert_raises(TypeError, EyelinkController, ec, output_dir=1)
    el = EyelinkController(ec, output_dir=temp_dir)
    assert_raises(TypeError, el.custom_calibration, 'blah')
    assert_raises(KeyError, el.custom_calibration, dict(me='hey'))
    assert_raises(ValueError, el.custom_calibration, dict(type='hey'))
    el.get_eye_position()
    x = el.wait_for_fix([-10000, -10000], max_wait=0.1)
    assert_true(x is False)
    assert el.eye_used
    el.start()
    el.stamp_trial_id(1)
    el.stamp_trial_id([1, 1])
    el.stamp_trial_id(['hello', 1, 12])
    assert_raises(ValueError, el.stamp_trial_id, [1, dict()])
    assert_raises(ValueError, el.stamp_trial_id, 'y' * 13)
    #el.calibrate()
    assert_raises(TypeError, el._message, 1)
    el.stop()
    el.save()  # auto-calls el.close()
    ec.close()
