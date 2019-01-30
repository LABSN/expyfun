from copy import deepcopy
from functools import partial

import numpy as np
from numpy.testing import assert_equal
import pytest
from unittest import SkipTest
from numpy.testing import assert_allclose

from expyfun import ExperimentController, wait_secs, visual
from expyfun._utils import (_TempDir, _hide_window, fake_button_press,
                            fake_mouse_click, requires_opengl21)
from expyfun.stimuli import get_tdt_rates
import sys
import os

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
        pytest.raises(ValueError, ec._convert_units, verts, 'deg', 'nothing')
        pytest.raises(RuntimeError, ec._convert_units, verts[0], 'deg', 'pix')


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
        assert (x in lines[1])
    assert ('flip' in lines[2])  # ec.__init__ ends with a flip
    assert ('stop' in lines[-1])  # last line is stop (from __exit__)
    outs = lines[1].strip().split('\t')
    assert (all(l1 == l2 for l1, l2 in zip(outs, ['timestamp',
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
    assert (np.all(ts[1:] >= ts[:-1]))


@pytest.mark.parametrize('ac, rd', ((None, None), ('tdt', 'tdt')))
@_hide_window
def test_ec(ac, rd):
    """Test EC methods."""
    if ac is None:
        # test type/value checking for audio_controller
        pytest.raises(TypeError, ExperimentController, *std_args,
                      audio_controller=1, stim_fs=44100, **std_kwargs)
        pytest.raises(ValueError, ExperimentController, *std_args,
                      audio_controller='foo', stim_fs=44100, **std_kwargs)
        pytest.raises(ValueError, ExperimentController, *std_args,
                      audio_controller=dict(TYPE='foo'), stim_fs=44100,
                      **std_kwargs)
        # monitor, etc.
        pytest.raises(TypeError, ExperimentController, *std_args,
                      monitor='foo', **std_kwargs)
        pytest.raises(KeyError, ExperimentController, *std_args,
                      monitor=dict(), **std_kwargs)
        pytest.raises(ValueError, ExperimentController, *std_args,
                      response_device='foo', **std_kwargs)
        std_kwargs.update(window_size=10.)
        pytest.raises(ValueError, ExperimentController, *std_args,
                      **std_kwargs)
        std_kwargs.update(window_size=(1, 1))
        pytest.raises(ValueError, ExperimentController, *std_args,
                      audio_controller='pyglet', response_device='tdt',
                      **std_kwargs)
        pytest.raises(ValueError, ExperimentController, *std_args,
                      audio_controller='pyglet', response_device='keyboard',
                      trigger_controller='tdt', **std_kwargs)

        # test type checking for 'session'
        std_kwargs['session'] = 1
        pytest.raises(TypeError, ExperimentController, *std_args,
                      audio_controller='pyglet', stim_fs=44100, **std_kwargs)
        std_kwargs['session'] = '01'

        # test value checking for trigger controller
        pytest.raises(ValueError, ExperimentController, *std_args,
                      audio_controller='pyglet', trigger_controller='foo',
                      stim_fs=44100, **std_kwargs)

        # test value checking for RMS checker
        pytest.raises(ValueError, ExperimentController, *std_args,
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
        pytest.raises(ValueError, ExperimentController, *std_args,
                      audio_controller=dict(TYPE=this_ac, TDT_MODEL='foo'),
                      **std_kwargs)
    for suppress in (True, False):
        with pytest.warns(None) as w:
            with ExperimentController(*std_args, audio_controller=this_ac,
                                      response_device=this_rd,
                                      trigger_controller=this_tc,
                                      stim_fs=100., suppress_resamp=suppress,
                                      **std_kwargs) as ec:
                pass
        assert len(w) == (1 if ac == 'tdt' else 0)
    with ExperimentController(*std_args, audio_controller=this_ac,
                              response_device=this_rd,
                              trigger_controller=this_tc,
                              stim_fs=this_fs, **std_kwargs) as ec:
        assert (ec.participant == std_kwargs['participant'])
        assert (ec.session == std_kwargs['session'])
        assert (ec.exp_name == std_args[0])
        stamp = ec.current_time
        ec.write_data_line('hello')
        ec.wait_until(stamp + 0.02)
        ec.screen_prompt('test', 0.01, 0, None)
        ec.screen_prompt('test', 0.01, 0, ['1'])
        ec.screen_prompt(['test', 'ing'], 0.01, 0, ['1'])
        ec.screen_prompt('test', 1e-3, click=True)
        pytest.raises(ValueError, ec.screen_prompt, 'foo', np.inf, 0, [])
        pytest.raises(TypeError, ec.screen_prompt, 3, 0.01, 0, None)
        assert_equal(ec.wait_one_press(0.01), (None, None))
        assert (ec.wait_one_press(0.01, timestamp=False) is None)
        assert_equal(ec.wait_for_presses(0.01), [])
        assert_equal(ec.wait_for_presses(0.01, timestamp=False), [])
        pytest.raises(ValueError, ec.get_presses)
        ec.listen_presses()
        assert_equal(ec.get_presses(), [])
        assert_equal(ec.get_presses(kind='presses'), [])
        pytest.raises(ValueError, ec.get_presses, kind='foo')
        if this_rd == 'tdt':
            # TDT does not have key release events, so should raise an
            # exception if asked for them:
            pytest.raises(RuntimeError, ec.get_presses, kind='releases')
            pytest.raises(RuntimeError, ec.get_presses, kind='both')
        else:
            assert_equal(ec.get_presses(kind='both'), [])
            assert_equal(ec.get_presses(kind='releases'), [])
        ec.set_noise_db(0)
        ec.set_stim_db(20)
        # test buffer data handling
        ec.set_rms_checking(None)
        ec.load_buffer([0, 0, 0, 0, 0, 0])
        pytest.raises(ValueError, ec.load_buffer, [0, 2, 0, 0, 0, 0])
        ec.load_buffer(np.zeros((100,)))
        ec.load_buffer(np.zeros((100, 1)))
        ec.load_buffer(np.zeros((100, 2)))
        ec.load_buffer(np.zeros((1, 100)))
        ec.load_buffer(np.zeros((2, 100)))
        data = np.zeros(int(5e6), np.float32)  # too long for TDT
        if this_fs == get_tdt_rates()['25k']:
            pytest.raises(RuntimeError, ec.load_buffer, data)
        else:
            ec.load_buffer(data)
        ec.load_buffer(np.zeros(2))
        del data
        pytest.raises(ValueError, ec.stamp_triggers, 'foo')
        pytest.raises(ValueError, ec.stamp_triggers, 0)
        pytest.raises(ValueError, ec.stamp_triggers, 3)
        pytest.raises(ValueError, ec.stamp_triggers, 1, check='foo')
        print(ec._tc)  # test __repr__
        if this_tc == 'dummy':
            assert_equal(ec._tc._trigger_list, [])
        ec.stamp_triggers(3, check='int4')
        ec.stamp_triggers(2)
        ec.stamp_triggers([2, 4, 8])
        if this_tc == 'dummy':
            assert_equal(ec._tc._trigger_list, [3, 2, 2, 4, 8])
            ec._tc._trigger_list = list()
        pytest.raises(ValueError, ec.load_buffer, np.zeros((100, 3)))
        pytest.raises(ValueError, ec.load_buffer, np.zeros((3, 100)))
        pytest.raises(ValueError, ec.load_buffer, np.zeros((1, 1, 1)))

        # test RMS checking
        pytest.raises(ValueError, ec.set_rms_checking, 'foo')
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
        ec.load_buffer(click)  # should pass
        with pytest.warns(UserWarning, match='exceeds stated'):
            ec.load_buffer(noise)
        ec.set_rms_checking('windowed')
        with pytest.warns(UserWarning, match='exceeds stated'):
            ec.load_buffer(click)
        with pytest.warns(UserWarning, match='exceeds stated'):
            ec.load_buffer(noise)

        ec.stop()
        ec.set_visible()
        ec.set_visible(False)
        ec.call_on_every_flip(partial(dummy_print, 'called start stimuli'))

        # Note: we put some wait_secs in here because otherwise the delay in
        # play start (e.g. for trigdel and onsetdel) can
        # mess things up! So we probably eventually should add
        # some safeguard against stopping too quickly after starting...

        #
        # First: identify_trial
        #
        noise = np.random.normal(scale=0.01, size=(int(ec.fs),))
        ec.load_buffer(noise)
        pytest.raises(RuntimeError, ec.start_stimulus)  # order violation
        assert (ec._playing is False)
        if this_tc == 'dummy':
            assert_equal(ec._tc._trigger_list, [])
        ec.start_stimulus(start_of_trial=False)         # should work
        if this_tc == 'dummy':
            assert_equal(ec._tc._trigger_list, [1])
        ec.wait_secs(0.05)
        assert (ec._playing is True)
        pytest.raises(RuntimeError, ec.trial_ok)        # order violation
        ec.stop()
        assert (ec._playing is False)
        # only binary for TTL
        pytest.raises(KeyError, ec.identify_trial, ec_id='foo')  # need ttl_id
        pytest.raises(TypeError, ec.identify_trial, ec_id='foo', ttl_id='bar')
        pytest.raises(ValueError, ec.identify_trial, ec_id='foo', ttl_id=[2])
        assert (ec._playing is False)
        if this_tc == 'dummy':
            ec._tc._trigger_list = list()
        ec.identify_trial(ec_id='foo', ttl_id=[0, 1])
        assert (ec._playing is False)
        #
        # Second: start_stimuli
        #
        pytest.raises(RuntimeError, ec.identify_trial, ec_id='foo', ttl_id=[0])
        assert (ec._playing is False)
        pytest.raises(RuntimeError, ec.trial_ok)        # order violation
        assert (ec._playing is False)
        ec.start_stimulus(flip=False, when=-1)
        if this_tc == 'dummy':
            assert_equal(ec._tc._trigger_list, [4, 8, 1])
        if ac != 'tdt':
            # dummy TDT version won't do this check properly, as
            # ec._ac._playing -> GetTagVal('playing') always gives False
            pytest.raises(RuntimeError, ec.play)  # already played, must stop
        ec.wait_secs(0.05)
        ec.stop()
        assert (ec._playing is False)
        #
        # Third: trial_ok
        #
        pytest.raises(RuntimeError, ec.start_stimulus)  # order violation
        pytest.raises(RuntimeError, ec.identify_trial)  # order violation
        ec.trial_ok()
        # double-check
        pytest.raises(RuntimeError, ec.start_stimulus)  # order violation
        ec.start_stimulus(start_of_trial=False)         # should work
        pytest.raises(RuntimeError, ec.trial_ok)        # order violation
        ec.wait_secs(0.05)
        ec.stop()
        assert (ec._playing is False)

        ec.flip(-np.inf)
        assert (ec._playing is False)
        ec.estimate_screen_fs()
        assert (ec._playing is False)
        ec.play()
        ec.wait_secs(0.05)
        assert (ec._playing is True)
        ec.call_on_every_flip(None)
        ec.call_on_next_flip(ec.start_noise())
        ec.wait_secs(0.05)
        ec.stop()
        assert (ec._playing is False)
        ec.start_stimulus(start_of_trial=False)
        ec.call_on_next_flip(ec.stop_noise)
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
def test_tdtpy_failure(ac=None, rd=None):
    """Test that failed TDTpy import raises ImportError."""
    try:
        from tdt.util import connect_rpcox  # noqa, analysis:ignore
    except ImportError:
        pass
    else:
        raise SkipTest('Cannot test TDT import failure')
    ac = dict(TYPE='tdt', TDT_MODEL='RP2')
    pytest.raises(ImportError, ExperimentController,
                  *std_args, audio_controller=ac, response_device='keyboard',
                  trigger_controller='tdt', stim_fs=100.,
                  suppress_resamp=True, **std_kwargs)


@_hide_window
def test_button_presses_and_window_size():
    """Test EC window_size=None and button press capture."""
    with ExperimentController(*std_args, audio_controller='pyglet',
                              response_device='keyboard', window_size=None,
                              output_dir=None, full_screen=False, session='01',
                              participant='foo', trigger_controller='dummy',
                              force_quit='escape', version='dev') as ec:
        ec.listen_presses()
        ec.get_presses()
        assert_equal(ec.get_presses(), [])

        fake_button_press(ec, '1', 0.5)
        assert_equal(ec.screen_prompt('press 1', live_keys=['1'],
                                      max_wait=1.5), '1')
        ec.listen_presses()
        assert_equal(ec.get_presses(), [])
        fake_button_press(ec, '1')
        assert_equal(ec.get_presses(timestamp=False), [('1',)])

        ec.listen_presses()
        fake_button_press(ec, '1')
        presses = ec.get_presses(timestamp=True, relative_to=0.2)
        assert_equal(len(presses), 1)
        assert_equal(len(presses[0]), 2)
        assert_equal(presses[0][0], '1')
        assert (isinstance(presses[0][1], float))

        ec.listen_presses()
        fake_button_press(ec, '1')
        presses = ec.get_presses(timestamp=True, relative_to=0.1,
                                 return_kinds=True)
        assert_equal(len(presses), 1)
        assert_equal(len(presses[0]), 3)
        assert_equal(presses[0][::2], ('1', 'press'))
        assert (isinstance(presses[0][1], float))

        ec.listen_presses()
        fake_button_press(ec, '1')
        presses = ec.get_presses(timestamp=False, return_kinds=True)
        assert_equal(presses, [('1', 'press')])

        ec.listen_presses()
        ec.screen_text('press 1 again')
        ec.flip()
        fake_button_press(ec, '1', 0.3)
        assert_equal(ec.wait_one_press(1.5, live_keys=[1])[0], '1')
        ec.screen_text('press 1 one last time')
        ec.flip()
        fake_button_press(ec, '1', 0.3)
        out = ec.wait_for_presses(1.5, live_keys=['1'], timestamp=False)
        assert_equal(out[0], '1')
        fake_button_press(ec, 'a', 0.3)
        fake_button_press(ec, 'return', 0.5)
        assert ec.text_input() == 'A'
        fake_button_press(ec, 'a', 0.3)
        fake_button_press(ec, 'space', 0.35)
        fake_button_press(ec, 'backspace', 0.4)
        fake_button_press(ec, 'comma', 0.45)
        fake_button_press(ec, 'return', 0.5)
        # XXX this fails on OSX travis for some reason
        if (os.getenv('TRAVIS', '').lower() != 'true' or
                sys.platform != 'darwin'):
            assert ec.text_input(all_caps=False) == 'a'


@_hide_window
@requires_opengl21
def test_mouse_clicks():
    """Test EC mouse click support."""
    with ExperimentController(*std_args, participant='foo', session='01',
                              output_dir=None, version='dev') as ec:
        rect = visual.Rectangle(ec, [0, 0, 2, 2])
        fake_mouse_click(ec, [1, 2], delay=0.3)
        assert_equal(ec.wait_for_click_on(rect, 1.5, timestamp=False)[0],
                     ('left', 1, 2))
        pytest.raises(TypeError, ec.wait_for_click_on, (rect, rect), 1.5)
        fake_mouse_click(ec, [2, 1], 'middle', delay=0.3)
        out = ec.wait_one_click(1.5, 0., ['middle'], timestamp=True)
        assert (out[3] < 1.5)
        assert_equal(out[:3], ('middle', 2, 1))
        fake_mouse_click(ec, [3, 2], 'left', delay=0.3)
        fake_mouse_click(ec, [4, 5], 'right', delay=0.3)
        out = ec.wait_for_clicks(1.5, timestamp=False)
        assert_equal(len(out), 2)
        assert (any(o == ('left', 3, 2) for o in out))
        assert (any(o == ('right', 4, 5) for o in out))
        out = ec.wait_for_clicks(0.1)
        assert_equal(len(out), 0)


@_hide_window
@requires_opengl21
def test_background_color():
    """Test setting background color"""
    with ExperimentController(*std_args, participant='foo', session='01',
                              output_dir=None, version='dev') as ec:
        ec.set_background_color('red')
        ss = ec.screenshot()[:, :, :3]
        red_mask = (ss == [255, 0, 0]).all(axis=-1)
        assert (red_mask.all())
        ec.set_background_color('white')
        ss = ec.screenshot()[:, :, :3]
        white_mask = (ss == [255] * 3).all(axis=-1)
        assert (white_mask.all())
        ec.flip()
        ec.set_background_color('0.5')
        visual.Rectangle(ec, [0, 0, 1, 1], fill_color='black').draw()
        ss = ec.screenshot()[:, :, :3]
        gray_mask = ((ss == [127] * 3).all(axis=-1) |
                     (ss == [128] * 3).all(axis=-1))
        assert (gray_mask.any())
        black_mask = (ss == [0] * 3).all(axis=-1)
        assert (black_mask.any())
        assert (np.logical_or(gray_mask, black_mask).all())


@_hide_window
def test_tdt_delay():
    """test the tdt_delay parameter"""
    with ExperimentController(*std_args,
                              audio_controller=dict(TYPE='tdt', TDT_DELAY=0),
                              **std_kwargs) as ec:
        assert_equal(ec._ac._used_params['TDT_DELAY'], 0)
    with ExperimentController(*std_args,
                              audio_controller=dict(TYPE='tdt', TDT_DELAY=1),
                              **std_kwargs) as ec:
        assert_equal(ec._ac._used_params['TDT_DELAY'], 1)
    pytest.raises(ValueError, ExperimentController, *std_args,
                  audio_controller=dict(TYPE='tdt', TDT_DELAY='foo'),
                  **std_kwargs)
    pytest.raises(OverflowError, ExperimentController, *std_args,
                  audio_controller=dict(TYPE='tdt', TDT_DELAY=np.inf),
                  **std_kwargs)
    pytest.raises(TypeError, ExperimentController, *std_args,
                  audio_controller=dict(TYPE='tdt', TDT_DELAY=np.ones(2)),
                  **std_kwargs)
    pytest.raises(ValueError, ExperimentController, *std_args,
                  audio_controller=dict(TYPE='tdt', TDT_DELAY=-1),
                  **std_kwargs)
