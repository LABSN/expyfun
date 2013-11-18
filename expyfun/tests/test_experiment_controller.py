import warnings
import numpy as np
from nose.tools import assert_raises, assert_true, assert_equal
from numpy.testing import assert_allclose, assert_array_equal

from expyfun import ExperimentController, wait_secs
from expyfun._utils import _TempDir, interactive_test, tdt_test

warnings.simplefilter('always')

temp_dir = _TempDir()
std_args = ['test']  # experiment name
std_kwargs = dict(output_dir=temp_dir, full_screen=False, window_size=(1, 1),
                  participant='foo', session='01', stim_db=0.0, noise_db=0.0,
                  stim_fs=48000)


def dummy_print(string):
    print string


def test_unit_conversions():
    """Test unit conversions
    """
    with ExperimentController(*std_args, **std_kwargs) as ec:
        verts = np.random.rand(2, 4)
        for to in ['norm', 'pix', 'deg']:
            for fro in ['norm', 'pix', 'deg']:
                v2 = ec._convert_units(verts, fro, to)
                v2 = ec._convert_units(v2, to, fro)
                assert_allclose(verts, v2)

        # test that degrees yield equiv. pixels in both directions
        verts = np.ones((2, 1))
        v0 = ec._convert_units(verts, 'deg', 'pix')
        verts = np.zeros((2, 1))
        v1 = ec._convert_units(verts, 'deg', 'pix')
        v2 = v0 - v1  # must check deviation from zero position
        assert_array_equal(v2[0], v2[1])


def test_no_output():
    """Test EC with no output
    """
    old_val = std_kwargs['output_dir']
    std_kwargs['output_dir'] = None
    try:
        with ExperimentController(*std_args, **std_kwargs) as ec:
            ec.write_data_line('hello')
    except:
        raise
    finally:
        std_kwargs['output_dir'] = old_val


def test_data_line():
    """Test writing of data lines
    """
    entries = [['foo'],
               ['bar', 'bar\tbar'],
               ['bar2', r'bar\tbar'],
               ['fb', None, -0.5]]
    # this is what should be written to the file for each one
    goal_vals = ['None', 'bar\\tbar', 'bar\\\\tbar', 'None']
    assert_equal(len(entries), len(goal_vals))

    with ExperimentController(*std_args, **std_kwargs) as ec:
        for ent in entries:
            ec.write_data_line(*ent)
        fname = ec._data_file.name
    with open(fname) as fid:
        lines = fid.readlines()
    # check the header
    assert_equal(len(lines), len(entries) + 3)
    assert_equal(lines[0][0], '#')  # first line is a comment
    for x in ['timestamp', 'event', 'value']:  # second line is col header
        assert_true(x in lines[1])
    assert_true('stop' in lines[-1])  # last line is stop (from __exit__)
    outs = lines[1].strip().split('\t')
    assert_true(all(l1 == l2 for l1, l2 in zip(outs, ['timestamp',
                                                      'event', 'value'])))
    # check the entries
    ts = []
    for line, ent, gv in zip(lines[2:], entries, goal_vals):
        outs = line.strip().split('\t')
        assert_equal(len(outs), 3)
        # check timestamping
        if len(ent) == 3 and ent[2] is not None:
            assert_true(outs[0] == str(ent[2]))
        else:
            ts.append(float(outs[0]))
        # check events
        assert_true(outs[1] == ent[0])
        # check values
        assert_true(outs[2] == gv)
    # make sure we got monotonically increasing timestamps
    ts = np.array(ts)
    assert_true(np.all(ts[1:] >= ts[:-1]))


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
                      audio_controller='pyo', **std_kwargs)
        std_kwargs['session'] = '01'

        # test value checking for trigger controller
        assert_raises(ValueError, ExperimentController, *std_args,
                      audio_controller='pyo', trigger_controller='foo',
                      **std_kwargs)

        # test value checking for RMS checker
        assert_raises(ValueError, ExperimentController, *std_args,
                      audio_controller='pyo', check_rms=True,
                      **std_kwargs)

        # run rest of test with audio_controller == 'pyo'
        ec = ExperimentController(*std_args, audio_controller='pyo',
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
    ec.screen_prompt(['test', 'ing'], 0.01, 0, ['1'])
    assert_raises(ValueError, ec.screen_prompt, 'foo', np.inf, 0, [])
    assert_raises(TypeError, ec.screen_prompt, 3, 0.01, 0, None)
    assert_equal(ec.wait_one_press(0.01), (None, None))
    assert_true(ec.wait_one_press(0.01, timestamp=False) is None)
    assert_equal(ec.wait_for_presses(0.01), [])
    assert_equal(ec.wait_for_presses(0.01, timestamp=False), [])
    assert_raises(ValueError, ec.get_presses)
    ec.listen_presses()
    assert_equal(ec.get_presses(), [])
    ec.clear_buffer()
    ec.set_noise_db(0)
    ec.set_stim_db(20)
    ec.draw_background_color('black')
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

    # test RMS checking
    assert_raises(ValueError, ec.set_rms_checking, 'foo')
    # click: RMS 0.0135, should pass 'fullfile' and fail 'windowed'
    click = np.zeros((ec.fs / 4,))  # 250 ms
    click[len(click) / 2] = 1.
    click[len(click) / 2 + 1] = -1.
    # noise: RMS 0.03, should fail both 'fullfile' and 'windowed'
    noise = np.random.normal(scale=0.03, size=(ec.fs / 4,))
    ec.set_rms_checking(None)
    ec.load_buffer(click)  # should go unchecked
    ec.load_buffer(noise)  # should go unchecked
    ec.set_rms_checking('wholefile')
    ec.load_buffer(click)  # should pass
    assert_raises(UserWarning, ec.load_buffer, noise)
    ec.set_rms_checking('windowed')
    assert_raises(UserWarning, ec.load_buffer, click)
    assert_raises(UserWarning, ec.load_buffer, noise)

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
    # test __repr__
    assert all([x in repr(ec) for x in ['foo', '"test"', '01']])
    ec.close()
    del ec


@interactive_test
def test_button_presses_and_window_size():
    """Test EC window_size=None and button press capture (press 1 thrice)
    """
    ec = ExperimentController(*std_args, audio_controller='pyo',
                              response_device='keyboard', window_size=None,
                              output_dir=temp_dir, full_screen=False,
                              participant='foo', session='01')
    assert_equal(ec.screen_prompt('press 1', live_keys=['1']), '1')
    ec.screen_text('press 1 again')
    assert_equal(ec.wait_one_press(live_keys=[1])[0], '1')
    ec.screen_text('press 1 one last time')
    out = ec.wait_for_presses(1.5, live_keys=['1'], timestamp=False)
    if len(out) > 0:
        assert_equal(out[0], '1')
    else:
        warnings.warn('press "1" faster next time')
    ec.close()
    del ec


def test_with_support():
    """Test EC 'with' statement support
    """
    with ExperimentController(*std_args, **std_kwargs) as ec:
        print ec
