import warnings
import numpy as np
from nose.tools import assert_raises, assert_true, assert_equal
from numpy.testing import assert_allclose
from copy import deepcopy

from expyfun import ExperimentController, wait_secs, visual
from expyfun._utils import (_TempDir, _hide_window, fake_button_press,
                            fake_mouse_click, requires_opengl21)
from expyfun.stimuli import get_tdt_rates
from functools import partial

warnings.simplefilter('always')

std_args = ['test']  # experiment name
std_kwargs = dict(output_dir=None, full_screen=False, window_size=(1, 1),
                  participant='foo', session='01', stim_db=0.0, noise_db=0.0,
                  verbose=True, version='dev')


def dummy_print(string):
    print(string)


@_hide_window
def test_unit_conversions():
    """Test unit conversions
    """
    for ws in [(2, 1), (1, 1)]:
        kwargs = deepcopy(std_kwargs)
        kwargs['stim_fs'] = 44100
        kwargs['window_size'] = ws
        with ExperimentController(*std_args, **kwargs) as ec:
            verts = np.random.rand(2, 4)
            for to in ['norm', 'pix', 'deg']:
                for fro in ['norm', 'pix', 'deg']:
                    print((ws, to, fro))
                    v2 = ec._convert_units(verts, fro, to)
                    v2 = ec._convert_units(v2, to, fro)
                    assert_allclose(verts, v2)

        # test that degrees yield equiv. pixels in both directions
        verts = np.ones((2, 1))
        v0 = ec._convert_units(verts, 'deg', 'pix')
        verts = np.zeros((2, 1))
        v1 = ec._convert_units(verts, 'deg', 'pix')
        v2 = v0 - v1  # must check deviation from zero position
        assert_allclose(v2[0], v2[1])
        assert_raises(ValueError, ec._convert_units, verts, 'deg', 'nothing')
        assert_raises(RuntimeError, ec._convert_units, verts[0], 'deg', 'pix')


@_hide_window
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
    temp_dir = _TempDir()
    these_kwargs = deepcopy(std_kwargs)
    these_kwargs['output_dir'] = temp_dir
    with ExperimentController(*std_args, stim_fs=44100, **these_kwargs) as ec:
        for ent in entries:
            ec.write_data_line(*ent)
        fname = ec._data_file.name
    with open(fname) as fid:
        lines = fid.readlines()
    # check the header
    assert_equal(len(lines), len(entries) + 4)  # header, colnames, flip, stop
    assert_equal(lines[0][0], '#')  # first line is a comment
    for x in ['timestamp', 'event', 'value']:  # second line is col header
        assert_true(x in lines[1])
    assert_true('flip' in lines[2])  # ec.__init__ ends with a flip
    assert_true('stop' in lines[-1])  # last line is stop (from __exit__)
    outs = lines[1].strip().split('\t')
    assert_true(all(l1 == l2 for l1, l2 in zip(outs, ['timestamp',
                                                      'event', 'value'])))
    # check the entries
    ts = []
    for line, ent, gv in zip(lines[3:], entries, goal_vals):
        outs = line.strip().split('\t')
        assert_equal(len(outs), 3)
        # check timestamping
        if len(ent) == 3 and ent[2] is not None:
            assert_equal(outs[0], str(ent[2]))
        else:
            ts.append(float(outs[0]))
        # check events
        assert_equal(outs[1], ent[0])
        # check values
        assert_equal(outs[2], gv)
    # make sure we got monotonically increasing timestamps
    ts = np.array(ts)
    assert_true(np.all(ts[1:] >= ts[:-1]))


@_hide_window
def test_tdt():
    """Test EC with TDT
    """
    test_ec('tdt', 'tdt')


@_hide_window
def test_ec(ac=None, rd=None):
    """Test EC methods
    """
    if ac is None:
        # test type/value checking for audio_controller
        assert_raises(TypeError, ExperimentController, *std_args,
                      audio_controller=1, stim_fs=44100, **std_kwargs)
        assert_raises(ValueError, ExperimentController, *std_args,
                      audio_controller='foo', stim_fs=44100, **std_kwargs)
        assert_raises(ValueError, ExperimentController, *std_args,
                      audio_controller=dict(TYPE='foo'), stim_fs=44100,
                      **std_kwargs)
        # monitor, etc.
        assert_raises(TypeError, ExperimentController, *std_args,
                      monitor='foo', **std_kwargs)
        assert_raises(KeyError, ExperimentController, *std_args,
                      monitor=dict(), **std_kwargs)
        assert_raises(ValueError, ExperimentController, *std_args,
                      response_device='foo', **std_kwargs)
        std_kwargs.update(window_size=10.)
        assert_raises(ValueError, ExperimentController, *std_args,
                      **std_kwargs)
        std_kwargs.update(window_size=(1, 1))
        assert_raises(ValueError, ExperimentController, *std_args,
                      audio_controller='pyglet', response_device='tdt',
                      **std_kwargs)
        assert_raises(ValueError, ExperimentController, *std_args,
                      audio_controller='pyglet', response_device='keyboard',
                      trigger_controller='tdt', **std_kwargs)

        # test type checking for 'session'
        std_kwargs['session'] = 1
        assert_raises(TypeError, ExperimentController, *std_args,
                      audio_controller='pyglet', stim_fs=44100, **std_kwargs)
        std_kwargs['session'] = '01'

        # test value checking for trigger controller
        assert_raises(ValueError, ExperimentController, *std_args,
                      audio_controller='pyglet', trigger_controller='foo',
                      stim_fs=44100, **std_kwargs)

        # test value checking for RMS checker
        assert_raises(ValueError, ExperimentController, *std_args,
                      audio_controller='pyglet', check_rms=True, stim_fs=44100,
                      **std_kwargs)

        # run rest of test with audio_controller == 'pyglet'
        this_ac = 'pyglet'
        this_rd = 'keyboard'
        this_tc = 'dummy'
        this_fs = 44100
    else:
        assert ac == 'tdt'
        # run rest of test with audio_controller == 'tdt'
        this_ac = ac
        this_rd = rd
        this_tc = ac
        this_fs = get_tdt_rates()['25k']
    with warnings.catch_warnings(record=True) as w:
        for suppress in (True, False):
            with ExperimentController(*std_args, audio_controller=this_ac,
                                      response_device=this_rd,
                                      trigger_controller=this_tc,
                                      stim_fs=100., suppress_resamp=suppress,
                                      **std_kwargs) as ec:
                pass
    warnings.simplefilter('ignore')  # ignore dummy TDT warning
    with ExperimentController(*std_args, audio_controller=this_ac,
                              response_device=this_rd,
                              trigger_controller=this_tc,
                              stim_fs=this_fs, **std_kwargs) as ec:
        warnings.simplefilter('always')
        assert_true(ec.participant == std_kwargs['participant'])
        assert_true(ec.session == std_kwargs['session'])
        assert_true(ec.exp_name == std_args[0])
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
        assert_equal(ec.get_presses(kind='presses'), [])
        assert_raises(ValueError, ec.get_presses, kind='foo')
        if this_rd == 'tdt':
            # TDT does not have key release events, so should raise an
            # exception if asked for them:
            assert_raises(RuntimeError, ec.get_presses, kind='releases')
            assert_raises(RuntimeError, ec.get_presses, kind='both')
        else:
            assert_equal(ec.get_presses(kind='both'), [])
            assert_equal(ec.get_presses(kind='releases'), [])
        ec.clear_buffer()
        ec.set_noise_db(0)
        ec.set_stim_db(20)
        # test buffer data handling
        ec.set_rms_checking(None)
        ec.load_buffer([0, 0, 0, 0, 0, 0])
        assert_raises(ValueError, ec.load_buffer, [0, 2, 0, 0, 0, 0])
        ec.load_buffer(np.zeros((100,)))
        ec.load_buffer(np.zeros((100, 1)))
        ec.load_buffer(np.zeros((100, 2)))
        ec.load_buffer(np.zeros((1, 100)))
        ec.load_buffer(np.zeros((2, 100)))
        data = np.empty(int(5e6), np.float32)  # too long for TDT
        if this_fs == get_tdt_rates()['25k']:
            assert_raises(RuntimeError, ec.load_buffer, data)
        else:
            ec.load_buffer(data)
        ec.load_buffer(np.zeros(2))
        del data
        assert_raises(ValueError, ec.stamp_triggers, 'foo')
        assert_raises(ValueError, ec.stamp_triggers, 0)
        assert_raises(ValueError, ec.stamp_triggers, 3)
        assert_raises(ValueError, ec.stamp_triggers, 1, check='foo')
        ec.stamp_triggers(3, check='int4')
        ec.stamp_triggers(2)
        ec.stamp_triggers([2, 4, 8])
        assert_raises(ValueError, ec.load_buffer, np.zeros((100, 3)))
        assert_raises(ValueError, ec.load_buffer, np.zeros((3, 100)))
        assert_raises(ValueError, ec.load_buffer, np.zeros((1, 1, 1)))

        # test RMS checking
        assert_raises(ValueError, ec.set_rms_checking, 'foo')
        # click: RMS 0.0135, should pass 'fullfile' and fail 'windowed'
        click = np.zeros((int(ec.fs / 4),))  # 250 ms
        click[len(click) // 2] = 1.
        click[len(click) // 2 + 1] = -1.
        # noise: RMS 0.03, should fail both 'fullfile' and 'windowed'
        noise = np.random.normal(scale=0.03, size=(int(ec.fs / 4),))
        ec.set_rms_checking(None)
        ec.load_buffer(click)  # should go unchecked
        ec.load_buffer(noise)  # should go unchecked
        ec.set_rms_checking('wholefile')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            ec.load_buffer(click)  # should pass
            assert_equal(len(w), 0)
            ec.load_buffer(noise)
            assert_equal(len(w), 1)
            ec.set_rms_checking('windowed')
            ec.load_buffer(click)
            assert_equal(len(w), 2)
            ec.load_buffer(noise)
            assert_equal(len(w), 3)

        ec.stop()
        ec.set_visible()
        ec.set_visible(False)
        ec.call_on_every_flip(partial(dummy_print, 'called start stimuli'))

        #
        # First: identify_trial
        #
        assert_raises(RuntimeError, ec.start_stimulus)  # order violation
        ec.start_stimulus(start_of_trial=False)         # should work
        assert_raises(RuntimeError, ec.trial_ok)        # order violation
        ec.stop()
        # only binary for TTL
        assert_raises(KeyError, ec.identify_trial, ec_id='foo')  # need ttl_id
        assert_raises(TypeError, ec.identify_trial, ec_id='foo', ttl_id='bar')
        assert_raises(ValueError, ec.identify_trial, ec_id='foo', ttl_id=[2])
        ec.identify_trial(ec_id='foo', ttl_id=[0, 1])
        #
        # Second: start_stimuli
        #
        assert_raises(RuntimeError, ec.identify_trial, ec_id='foo', ttl_id=[0])
        assert_raises(RuntimeError, ec.trial_ok)        # order violation
        ec.start_stimulus(flip=False, when=-1)
        assert_raises(RuntimeError, ec.play)  # already played, must stop
        ec.stop()
        #
        # Third: trial_ok
        #
        assert_raises(RuntimeError, ec.start_stimulus)  # order violation
        assert_raises(RuntimeError, ec.identify_trial)  # order violation
        ec.trial_ok()
        # double-check
        assert_raises(RuntimeError, ec.start_stimulus)  # order violation
        ec.start_stimulus(start_of_trial=False)         # should work
        assert_raises(RuntimeError, ec.trial_ok)        # order violation
        ec.stop()

        ec.flip(-np.inf)
        ec.estimate_screen_fs()
        ec.play()
        ec.call_on_every_flip(None)
        ec.call_on_next_flip(ec.start_noise())
        ec.stop()
        ec.start_stimulus(start_of_trial=False)
        ec.call_on_next_flip(ec.stop_noise())
        ec.stop()
        ec.start_stimulus(start_of_trial=False)
        ec.get_mouse_position()
        ec.listen_clicks()
        ec.get_clicks()
        ec.toggle_cursor(False)
        ec.toggle_cursor(True, True)
        ec.wait_secs(0.001)
        print(ec.id_types)
        print(ec.stim_db)
        print(ec.noise_db)
        print(ec.on_next_flip_functions)
        print(ec.on_every_flip_functions)
        print(ec.window)
        data = ec.screenshot()
        assert_allclose(data.shape[:2], std_kwargs['window_size'])
        print(ec.fs)  # test fs support
        wait_secs(0.01)
        test_pix = (11.3, 0.5, 110003)
        print(test_pix)
        # test __repr__
        assert all([x in repr(ec) for x in ['foo', '"test"', '01']])
    del ec


@_hide_window
def test_button_presses_and_window_size():
    """Test EC window_size=None and button press capture
    """
    warnings.simplefilter('ignore')  # esc as quit key
    with ExperimentController(*std_args, audio_controller='pyglet',
                              response_device='keyboard', window_size=None,
                              output_dir=None, full_screen=False,
                              participant='foo', session='01',
                              force_quit='escape', version='dev') as ec:
        warnings.simplefilter('always')
        fake_button_press(ec, '1', 0.3)
        assert_equal(ec.screen_prompt('press 1', live_keys=['1'],
                                      max_wait=1.5), ('1', 'press'))
        ec.screen_text('press 1 again')
        ec.flip()
        fake_button_press(ec, '1', 0.3)
        assert_equal(ec.wait_one_press(1.5, live_keys=[1])[0], '1')
        ec.screen_text('press 1 one last time')
        ec.flip()
        fake_button_press(ec, '1', 0.3)
        out = ec.wait_for_presses(1.5, live_keys=['1'], timestamp=False)
        assert_equal(out[0], ('1', 'press'))


@_hide_window
@requires_opengl21
def test_mouse_clicks():
    """Test EC mouse click support
    """
    with ExperimentController(*std_args, participant='foo', session='01',
                              output_dir=None, version='dev') as ec:
        rect = visual.Rectangle(ec, [0, 0, 2, 2])
        fake_mouse_click(ec, [1, 2], delay=0.3)
        assert_equal(ec.wait_for_click_on(rect, 1.5, timestamp=False)[0],
                     ('left', 1, 2))
        assert_raises(TypeError, ec.wait_for_click_on, (rect, rect), 1.5)
        fake_mouse_click(ec, [2, 1], 'middle', delay=0.3)
        out = ec.wait_one_click(1.5, 0., ['middle'], timestamp=True)
        assert_true(out[3] < 1.5)
        assert_equal(out[:3], ('middle', 2, 1))
        fake_mouse_click(ec, [3, 2], 'left', delay=0.3)
        fake_mouse_click(ec, [4, 5], 'right', delay=0.3)
        out = ec.wait_for_clicks(1.5, timestamp=False)
        assert_equal(len(out), 2)
        assert_true(any(o == ('left', 3, 2) for o in out))
        assert_true(any(o == ('right', 4, 5) for o in out))
        out = ec.wait_for_clicks(0.1)
        assert_equal(len(out), 0)


@_hide_window
@requires_opengl21
def test_background_color():
    """Test setting background color"""
    with ExperimentController(*std_args, participant='foo', session='01',
                              output_dir=None, version='dev') as ec:
        ec.set_background_color('red')
        ec.screen_text('red', color='white')
        ss = ec.screenshot()[:, :, :3]
        white_mask = (ss == [255] * 3)
        assert_true(white_mask.any())
        red_mask = (ss == [255, 0, 0])
        assert_true(red_mask.any())
        assert_true(np.logical_or(white_mask, red_mask).all())
        ec.flip()

        ec.set_background_color('0.5')
        visual.Rectangle(ec, [0, 0, 1, 1], fill_color='black').draw()
        ss = ec.screenshot()[:, :, :3]
        gray_mask = (ss == [127] * 3) | (ss == [128] * 3)
        assert_true(gray_mask.any())
        black_mask = (ss == [0] * 3)
        assert_true(black_mask.any())
        assert_true(np.logical_or(gray_mask, black_mask).all())
