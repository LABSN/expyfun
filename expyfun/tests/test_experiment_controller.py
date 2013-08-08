import numpy as np
from nose.tools import assert_raises

from expyfun import ExperimentController
from expyfun.utils import _TempDir

temp_dir = _TempDir()
std_args = ['test']
std_kwargs = dict(output_dir=temp_dir, full_screen=False, window_size=(1, 1),
                  participant='foo', session='01')


def dummy_print(string):
    print string


def test_experiment_init():
    """Test experiment methods
    """
    ec = ExperimentController(*std_args, **std_kwargs)
    ec.screen_prompt('test', 0, 0.01)
    ec.add_data_line(dict(me='hello'))
    assert_raises(ValueError, ec.screen_prompt, 'foo', np.inf, 0, None)
    ec.clear_screen()
    ec.get_press(0.01)
    ec.get_presses(0.01)
    ec.clear_buffer()
    ec.set_noise_amp(0)
    ec.set_stim_amp(20)
    ec.load_buffer(np.zeros((100,)))
    #ec.load_buffer(np.zeros((100, 1))) XXX FAILS
    ec.load_buffer(np.zeros((100, 2)))
    ec.stop_reset()
    ec.call_on_flip_and_play(None)
    print ec.fs
    ec.flip_and_play()
    ec.wait_secs(0.01)
    ec.call_on_flip_and_play(dummy_print, 'hello')
    ec.flip_and_play()
    #ec.load_buffer(np.zeros((1, 100))) XXX FAILS WITHOUT INFORMATIVE MESSAGE
    # test __repr__
    assert all([x in repr(ec) for x in ['foo', '"test"', '01']])
    ec.init_trial()
    ec.close()


def test_with_support():
    """Test experiment 'with' statement support
    """
    with ExperimentController(*std_args, **std_kwargs) as ec:
        print ec
