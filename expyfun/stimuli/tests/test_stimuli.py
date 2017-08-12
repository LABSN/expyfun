# -*- coding: utf-8 -*-

import numpy as np
import warnings
from nose.tools import assert_raises, assert_equal, assert_true
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_allclose)
from scipy.signal import butter, lfilter

from expyfun._utils import _TempDir, requires_lib, _hide_window
from expyfun.stimuli import (rms, play_sound, convolve_hrtf, window_edges,
                             vocode, texture_ERB, crm_info, crm_prepare_corpus,
                             crm_sentence, crm_response_menu, CRMPreload,
                             add_pad)

warnings.simplefilter('always')

tempdir = _TempDir()


def test_textures():
    """Test stimulus textures."""
    texture_ERB()  # smoke test
    assert_raises(TypeError, texture_ERB, seq='foo')
    assert_raises(ValueError, texture_ERB, seq=('foo',))
    with warnings.catch_warnings(record=True) as w:
        x = texture_ERB(freq_lims=(200, 500))
    assert_true(any('less than' in str(ww.message).lower() for ww in w))
    assert_allclose(len(x) / 24414., 4., rtol=1e-5)


@requires_lib('h5py')
def test_hrtf_convolution():
    """Test HRTF convolution."""
    data = np.random.randn(2, 10000)
    assert_raises(ValueError, convolve_hrtf, data, 44100, 0, interp=False)
    data = data[0]
    assert_raises(ValueError, convolve_hrtf, data, 44100, 0.5, interp=False)
    assert_raises(ValueError, convolve_hrtf, data, 44100, 0,
                  source='foo', interp=False)
    assert_raises(ValueError, convolve_hrtf, data, 44100, 90.5, interp=True)
    assert_raises(ValueError, convolve_hrtf, data, 44100, 0, interp='foo')
    # invalid angle when interp=False
    for interp in [True, False]:
        for source in ['barb', 'cipic']:
            if interp and source == 'barb':
                # raise an error when trying to interp with 'barb'
                assert_raises(ValueError, convolve_hrtf, data, 44100, 2.5,
                              source=source, interp=interp)
            else:
                out = convolve_hrtf(data, 44100, 0, source=source,
                                    interp=interp)
                out_2 = convolve_hrtf(data, 24414, 0, source=source,
                                      interp=interp)
                assert_equal(out.ndim, 2)
                assert_equal(out.shape[0], 2)
                assert_true(out.shape[1] > data.size)
                assert_true(out_2.shape[1] < out.shape[1])
                if interp:
                    out_3 = convolve_hrtf(data, 44100, 2.5, source=source,
                                          interp=interp)
                    out_4 = convolve_hrtf(data, 44100, -2.5, source=source,
                                          interp=interp)
                    assert_equal(out_3.ndim, 2)
                    assert_equal(out_4.ndim, 2)
                    # ensure that, at least for zero degrees, it's close
                out = convolve_hrtf(data, 44100, 0, source=source,
                                    interp=interp)[:, 1024:-1024]
                assert_allclose(np.mean(rms(out)), rms(data), rtol=1e-1)
                out = convolve_hrtf(data, 44100, -90, source=source,
                                    interp=interp)
                rmss = rms(out)
                assert_true(rmss[0] > 4 * rmss[1])


@_hide_window  # will only work if Pyglet windowing works
def test_play_sound():
    """Test playing a sound."""
    data = np.zeros((2, 100))
    play_sound(data).stop()
    play_sound(data[0], norm=False, wait=True)
    assert_raises(ValueError, play_sound, data[:, :, np.newaxis])
    # Make sure Pyglet can handle a lot of sounds
    for _ in range(100):
        snd = play_sound(data)
        # we manually stop and delete here, because we don't want to
        # have to wait for our Timer instances to get around to doing
        # it... this also checks to make sure calling `delete()` more
        # than once is okay (it is).
        snd.stop()
        snd.delete()


def test_window_edges():
    """Test windowing signal edges."""
    sig = np.ones((2, 1000))
    fs = 44100
    assert_raises(ValueError, window_edges, sig, fs, window='foo')  # bad win
    assert_raises(RuntimeError, window_edges, sig, fs, dur=1.0)  # too long
    assert_raises(ValueError, window_edges, sig, fs, edges='foo')  # bad type
    x = window_edges(sig, fs, edges='leading')
    y = window_edges(sig, fs, edges='trailing')
    z = window_edges(sig, fs)
    assert_true(np.all(x[:, 0] < 1))  # make sure we actually reduced amp
    assert_true(np.all(x[:, -1] == 1))
    assert_true(np.all(y[:, 0] == 1))
    assert_true(np.all(y[:, -1] < 1))
    assert_allclose(x + y, z + 1)


def _voc_similarity(orig, voc):
    """Quantify envelope similiarity after vocoding."""
    return np.correlate(orig, voc, mode='full').max()


def test_vocoder():
    """Test noise, tone, and click vocoding."""
    data = np.random.randn(10000)
    env = np.random.randn(10000)
    b, a = butter(4, 0.001, 'lowpass')
    data *= lfilter(b, a, env)
    # bad limits
    assert_raises(ValueError, vocode, data, 44100, freq_lims=(200, 30000))
    # bad mode
    assert_raises(ValueError, vocode, data, 44100, mode='foo')
    # bad seed
    assert_raises(TypeError, vocode, data, 44100, seed='foo')
    assert_raises(ValueError, vocode, data, 44100, scale='foo')
    voc1 = vocode(data, 20000, mode='noise', scale='log')
    voc2 = vocode(data, 20000, mode='tone', order=4, seed=0, scale='hz')
    voc3 = vocode(data, 20000, mode='poisson', seed=np.random.RandomState(123))
    # XXX This is about the best we can do for now...
    assert_array_equal(voc1.shape, data.shape)
    assert_array_equal(voc2.shape, data.shape)
    assert_array_equal(voc3.shape, data.shape)


def test_rms():
    """Test RMS calculation."""
    # Test a couple trivial things we know
    sin = np.sin(2 * np.pi * 1000 * np.arange(10000, dtype=float) / 10000.)
    assert_array_almost_equal(rms(sin), 1. / np.sqrt(2))
    assert_array_almost_equal(rms(np.ones((100, 2)) * 2, 0), [2, 2])


def test_crm():
    """Test CRM Corpus functions."""
    fs = 40000  # native rate, to avoid large resampling delay in testing
    crm_info()

    # corpus prep
    talkers = [dict(sex='f', talker_num=0)]
    crm_prepare_corpus(fs, talker_list=talkers, n_jobs=np.inf)
    crm_prepare_corpus(fs, talker_list=talkers, overwrite=True)
    # no overwrite
    assert_raises(RuntimeError, crm_prepare_corpus, fs)

    # load sentence from hard drive
    crm_sentence(fs, 'f', 0, 0, 0, 0, 0, ramp_dur=0)
    crm_sentence(fs, '1', '0', 'charlie', 'red', '5', stereo=True)
    # bad value requested
    assert_raises(ValueError, crm_sentence, fs, 1, 0, 0, 'periwinkle', 0)
    # unprepared talker
    assert_raises(RuntimeError, crm_sentence, fs, 'm', 0, 0, 0, 0)
    # unprepared sampling rate
    assert_raises(RuntimeError, crm_sentence, fs + 1, 0, 0, 0, 0, 0)

    # CRMPreload class
    crm = CRMPreload(fs)
    # unprepared sampling rate
    assert_raises(RuntimeError, CRMPreload, fs + 1)
    crm.sentence(fs, 'f', 0, 0, 0, 0, 0, ramp_dur=0)
    crm.sentence(fs, '1', '0', 'charlie', 'red', '5', stereo=True)
    # bad value requested
    assert_raises(ValueError, crm.sentence, fs, 1, 0, 0, 'periwinkle', 0)
    # unprepared talker
    assert_raises(RuntimeError, crm.sentence, fs, 'm', 0, 0, 0, 0)

    # CRM response function


    # add_pad
    x1 = np.zeros(10)
    x2 = np.ones((2, 5))
    x = add_pad(x1, x2)
    assert_true(np.sum(x[..., -1] == 0))
    x = add_pad(x1, x2, 'center')
    assert_true(np.sum(x[..., -1] == 0) and np.sum(x[..., 0] == 0))
    x = add_pad(x1, x2, 'end')
    assert_true(np.sum(x[..., 0] == 0))