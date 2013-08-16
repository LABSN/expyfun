import numpy as np
from nose.tools import assert_raises

from expyfun import ExperimentController, wait_secs
from expyfun.utils import _TempDir, interactive_test

temp_dir = _TempDir()
std_args = ['test']  # experiment name
std_kwargs = dict(output_dir=temp_dir, full_screen=False, window_size=(1, 1),
                  participant='foo', session='01')


def dummy_print(string):
    print string


def test_stamping():
    """Test EC stamping support"""
    ec = ExperimentController(*std_args, **std_kwargs)
    ec.stamp_triggers([1, 2])
    ec.close()


def test_experiment_init():
    """Test EC methods
    """
    assert_raises(TypeError, ExperimentController, audio_controller=1,
                  *std_args, **std_kwargs)

    assert_raises(ValueError, ExperimentController, audio_controller='foo',
                  *std_args, **std_kwargs)

    assert_raises(ValueError, ExperimentController,
                  audio_controller=dict(TYPE='foo'), *std_args, **std_kwargs)

    assert_raises(TypeError, ExperimentController, *std_args, session=1,
                  participant='foo', output_dir=temp_dir, full_screen=False,
                  window_size=(1, 1))

    ec = ExperimentController(*std_args, **std_kwargs)
    ec.init_trial()
    ec.add_data_line(dict(me='hello'))
    ec.screen_prompt('test', 0.01, 0)
    ec.screen_prompt('test', 0.01, 0, None)
    assert_raises(ValueError, ec.screen_prompt, 'foo', np.inf, 0, None)
    ec.clear_screen()
    ec.get_press(0.01)
    ec.get_presses(0.01)
    ec.clear_buffer()
    ec.set_noise_amp(0)
    ec.set_stim_amp(20)
    ec.load_buffer(np.zeros((100,)))
    ec.load_buffer(np.zeros((100, 1)))
    ec.load_buffer(np.zeros((100, 2)))
    ec.stop()
    ec.call_on_flip_and_play(None)
    print ec.fs  # test fs support
    ec.flip_and_play()
    wait_secs(0.01)
    ec.call_on_flip_and_play(dummy_print, 'called on flip and play')
    ec.flip_and_play()
    assert_raises(ValueError, ec.load_buffer, np.zeros((1, 100)))
    # test __repr__
    assert all([x in repr(ec) for x in ['foo', '"test"', '01']])
    ec.close()


@interactive_test
def test_button_presses_and_window_size():
    """Test EC window_size=None and button press capture (press 1 thrice)
    """
    ec = ExperimentController(*std_args, audio_controller='psychopy',
                              response_device='keyboard', window_size=None,
                              output_dir=temp_dir, full_screen=False,
                              participant='foo', session='01')
    ec.screen_text('press 1 thrice')
    pressed = []
    while len(pressed) < 3:
        pressed.append(ec.get_press(live_keys=['1'])[0])
    assert pressed == ['1', '1', '1']
    ec.win.close()
    del ec


def test_with_support():
    """Test EC 'with' statement support
    """
    with ExperimentController(*std_args, **std_kwargs) as ec:
        print ec
