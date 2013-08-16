from nose.tools import assert_raises, assert_true
from expyfun import EyelinkController, ExperimentController
from expyfun.utils import _TempDir

from expyfun.utils import requires_pylink

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
    x = el.hold_Fix([-10000, -10000], 0.1)
    assert_true(x is False)
    x = el.wait_for_fix([-10000, -10000], max_wait=0.1)
    assert_true(x is False)
    el.eye_used()
    el.start()
    #el.calibrate()
    assert_raises(TypeError, el.message, 1)
    el.stop()
    el.save()  # auto-calls el.close()
    ec.close()
