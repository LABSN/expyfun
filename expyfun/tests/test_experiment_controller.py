import numpy as np
from nose.tools import assert_raises
from numpy.testing import assert_allclose

from expyfun import ExperimentController, wait_secs
from expyfun.utils import _TempDir, interactive_test, tdt_test

temp_dir = _TempDir()
std_args = ['test']  # experiment name
std_kwargs = dict(output_dir=temp_dir, full_screen=False, window_size=(1, 1),
                  participant='foo', session='01', stim_db=0.0, noise_db=0.0)


def dummy_print(string):
    print string


def test_stamping():
    """Test EC stamping support"""
    ec = ExperimentController(*std_args, **std_kwargs)
    ec.stamp_triggers([1, 2])
    ec.close()


@tdt_test
def test_tdt():
    """Test EC with TDT if possible
    """
    test_ec('tdt')


def test_ec(ac=None):
    """Test EC methods
    """
    if ac is None:
        # test type/value checking for audio_controller
        assert_raises(TypeError, ExperimentController, *std_args,
                      audio_controller=1, **std_kwargs)
        assert_raises(ValueError, ExperimentController, *std_args,
                      audio_controller='foo', **std_kwargs)
        assert_raises(ValueError, ExperimentController, *std_args,
                      audio_controller=dict(TYPE='foo'), **std_kwargs)

        # test type checking for 'session'
        std_kwargs['session'] = 1
        assert_raises(TypeError, ExperimentController, *std_args,
                      audio_controller='psychopy', **std_kwargs)
        std_kwargs['session'] = '01'

        # test value checking for trigger controller
        assert_raises(ValueError, ExperimentController, *std_args,
                      audio_controller='psychopy', trigger_controller='foo',
                      **std_kwargs)

        #run rest of test with audio_controller == 'psychopy'
        ec = ExperimentController(*std_args, audio_controller='psychopy',
                                  **std_kwargs)

    else:
        # run rest of test with audio_controller == 'tdt'
        ec = ExperimentController(*std_args, audio_controller=ac,
                                  **std_kwargs)

    stamp = ec.current_time
    ec.write_data_line('hello')
    ec.wait_until(stamp + 0.02)
    ec.screen_prompt('test', 0.01, 0, None)
    ec.screen_prompt('test', 0.01, 0, ['1'])
    assert_raises(ValueError, ec.screen_prompt, 'foo', np.inf, 0, [])
    ec.clear_screen()
    assert ec.wait_one_press(0.01) == (None, None)
    assert ec.wait_one_press(0.01, timestamp=False) is None
    assert ec.wait_for_presses(0.01) == []
    assert ec.wait_for_presses(0.01, timestamp=False) == []
    assert_raises(ValueError, ec.get_presses)
    ec.listen_presses()
    assert ec.get_presses() == []
    ec.clear_buffer()
    ec.set_noise_db(0)
    ec.set_stim_db(20)
    # test buffer data handling
    ec.load_buffer([0, 0, 0, 0, 0, 0])
    assert_raises(ValueError, ec.load_buffer, [0, 2, 0, 0, 0, 0])
    ec.load_buffer(np.zeros((100,)))
    ec.load_buffer(np.zeros((100, 1)))
    ec.load_buffer(np.zeros((100, 2)))
    ec.load_buffer(np.zeros((1, 100)))
    ec.load_buffer(np.zeros((2, 100)))
    assert_raises(ValueError, ec.load_buffer, np.zeros((100, 3)))
    assert_raises(ValueError, ec.load_buffer, np.zeros((3, 100)))
    assert_raises(ValueError, ec.load_buffer, np.zeros((1, 1, 1)))
    ec.stop()
    ec.call_on_every_flip(dummy_print, 'called on flip and play')
    ec.flip_and_play()
    ec.flip()
    ec.call_on_every_flip(None)
    ec.call_on_next_flip(ec.start_noise())
    ec.flip_and_play()
    ec.call_on_next_flip(ec.stop_noise())
    ec.flip_and_play()
    print ec.fs  # test fs support
    wait_secs(0.01)
    test_pix = (11.3, 0.5, 110003)
    print test_pix
    assert_allclose(test_pix, ec.deg2pix(ec.pix2deg(test_pix)))
    # test __repr__
    assert all([x in repr(ec) for x in ['foo', '"test"', '01']])
    ec.close()
    del ec


@interactive_test
def test_button_presses_and_window_size():
    """Test EC window_size=None and button press capture (press 1 thrice)
    """
    ec = ExperimentController(*std_args, audio_controller='psychopy',
                              response_device='keyboard', window_size=None,
                              output_dir=temp_dir, full_screen=False,
                              participant='foo', session='01')
    assert ec.screen_prompt('press 1', live_keys=['1']) == '1'
    ec.screen_text('press 1 again')
    assert ec.wait_one_press(live_keys=[1])[0] == '1'
    ec.screen_text('press 1 one last time')
    assert ec.wait_for_presses(0.5, live_keys=['1'], timestamp=False)[0] == '1'
    ec.close()
    del ec


def test_with_support():
    """Test EC 'with' statement support
    """
    with ExperimentController(*std_args, **std_kwargs) as ec:
        print ec
