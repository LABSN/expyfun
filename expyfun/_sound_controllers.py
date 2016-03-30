"""Hardware interfaces for sound output"""

# Authors: Dan McCloy <drmccloy@uw.edu>
#          Eric Larson <larsoner@uw.edu>
#
# License: BSD (3-clause)

import numpy as np
from scipy import fftpack
import sys
import os
import pyglet

_use_silent = (os.getenv('_EXPYFUN_SILENT', '') == 'true')
_opts_dict = dict(linux2=('pulse',),
                  win32=('directsound',),
                  darwin=('openal',))
_opts_dict['linux'] = _opts_dict['linux2']  # new name on Py3k
_driver = _opts_dict[sys.platform] if not _use_silent else ('silent',)

pyglet.options['audio'] = _driver

# these must follow the above option setting, so PEP8 complains
#try:
from pyglet.media import Player, AudioFormat, SourceGroup  # noqa
#except ImportError:
#    from pyglet.media import Player, AudioFormat
#    from mock import Mock
#    SourceGroup = Mock()  # noqa
try:
    from pyglet.media import StaticMemorySource
except ImportError:
    from pyglet.media.sources.base import StaticMemorySource  # noqa

from ._utils import logger, flush_logger  # noqa


def _check_pyglet_audio():
    if pyglet.media.get_audio_driver() is None:
        raise SystemError('pyglet audio ("%s") could not be initialized'
                          % pyglet.options['audio'][0])


class SoundPlayer(Player):
    def __init__(self, data, fs, loop=False):
        assert AudioFormat is not None
        super(SoundPlayer, self).__init__()
        _check_pyglet_audio()
        sms = _as_static(data, fs)
        group = SourceGroup(sms.audio_format, None)
        group.loop = bool(loop)
        group.queue(sms)
        self.queue(group)

    def stop(self):
        self.pause()
        self.seek(0.)


class PygletSoundController(object):
    """Use pyglet audio capabilities"""
    def __init__(self, ec, stim_fs):
        logger.info('Expyfun: Setting up Pyglet audio')
        assert AudioFormat is not None
        self.fs = stim_fs

        # Need to generate at RMS=1 to match TDT circuit
        noise = np.random.normal(0, 1.0, int(self.fs * 15.))  # 15 secs

        # Low-pass if necessary
        if stim_fs < self.fs:
            # note we can use cheap DFT method here b/c
            # circular convolution won't matter for AWGN (yay!)
            freqs = fftpack.fftfreq(len(noise), 1. / self.fs)
            noise = fftpack.fft(noise)
            noise[np.abs(freqs) > stim_fs / 2.] = 0.0
            noise = np.real(fftpack.ifft(noise))

        # ensure true RMS of 1.0 (DFT method also lowers RMS, compensate here)
        noise = noise / np.sqrt(np.mean(noise * noise))
        self.noise_array = np.array((noise, -1.0 * noise))
        self.noise = SoundPlayer(self.noise_array, self.fs, loop=True)
        self._noise_playing = False
        self.audio = SoundPlayer(np.zeros((2, 1)), self.fs)
        self.ec = ec
        flush_logger()

    def start_noise(self):
        if not self._noise_playing:
            self.noise.play()
            self._noise_playing = True

    def stop_noise(self):
        if self._noise_playing:
            self.noise.stop()
            self._noise_playing = False

    def clear_buffer(self):
        self.audio.delete()
        self.audio = SoundPlayer(np.zeros((2, 1)), self.fs)

    def load_buffer(self, samples):
        self.audio.delete()
        self.audio = SoundPlayer(samples.T, self.fs)

    def play(self):
        self.audio.play()
        self.ec._stamp_ttl_triggers([1])

    def stop(self):
        self.audio.stop()

    def set_noise_level(self, level):
        new_noise = SoundPlayer(self.noise_array * level, self.fs, loop=True)
        if self._noise_playing:
            self.stop_noise()
            self.noise.delete()
            self.noise = new_noise
            self.start_noise()
        else:
            self.noise = new_noise

    def halt(self):
        self.stop()
        self.stop_noise()
        # cleanup pyglet instances
        self.audio.delete()
        self.noise.delete()


def _as_static(data, fs):
    """Helper to get data into the Pyglet audio format"""
    fs = int(fs)
    if data.ndim not in (1, 2):
        raise ValueError('Data must have one or two dimensions')
    n_ch = data.shape[0] if data.ndim == 2 else 1
    audio_format = AudioFormat(channels=n_ch, sample_size=16,
                               sample_rate=fs)
    data = data.T.ravel('C')
    data[data < -1] = -1
    data[data > 1] = 1
    data = (data * (2 ** 15)).astype('int16').tostring()
    return StaticMemorySourceFixed(data, audio_format)


class StaticMemorySourceFixed(StaticMemorySource):
    """Stupid class to fix old Pyglet bug"""
    def _get_queue_source(self):
        return self
