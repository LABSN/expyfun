"""Hardware interfaces for sound output"""

# Authors: Dan McCloy <drmccloy@uw.edu>
#          Eric Larson <larsoner@uw.edu>
#
# License: BSD (3-clause)

import numpy as np
from scipy import fftpack
import sys
import pyglet

from ._utils import logger, flush_logger


from pyglet.media import StreamingSource, AudioFormat, AudioData, Player

_opts_dict = dict(linux2=('pulse',),
                  win32=('directsound',))
pyglet.options['audio'] = _opts_dict.get(sys.platform, ('openal',))


class SoundSource(StreamingSource):
    def __init__(self, data, fs):
        _check_pyglet_audio()
        fs = int(fs)
        assert data.ndim == 2
        assert data.shape[0] == 2
        data = np.ascontiguousarray(data.T).ravel()
        data = (np.clip(data, -1, 1) * (2 ** 15)).astype(np.int16).tostring()
        self._len = len(data)
        self._data = data
        self.audio_format = AudioFormat(channels=2, sample_size=16,
                                        sample_rate=fs)
        self._duration = self._len / self.audio_format.bytes_per_second
        self._offset = 0

    def get_audio_data(self, n_bytes):
        n_bytes = min(n_bytes, self._len - self._offset)
        if not n_bytes:
            return None

        data = self._data[self._offset:self._offset + n_bytes]
        timestamp = float(self._offset) / self.audio_format.bytes_per_second
        duration = float(n_bytes) / self.audio_format.bytes_per_second
        self._offset += len(data)
        return AudioData(data, len(data), timestamp, duration, [])

    def seek(self, timestamp):
        offset = int(timestamp * self.audio_format.bytes_per_second)
        offset = min(max(offset, 0), self._len)
        self._offset = offset

    def stop(self):
        # Pyglet doesn't provide this functionality, but this should work...
        self._offset = self._len


class SoundPlayer(Player):
    def __init__(self, data, fs, loop=False):
        Player.__init__(self)
        snd = SoundSource(data, fs)
        self.queue(snd)
        self.eos_action = self.EOS_LOOP if loop else self.EOS_PAUSE

    def stop(self):
        self.pause()
        self.seek(0.)


def _check_pyglet_audio():
    if pyglet.media.get_audio_driver() is None:
        raise SystemError('pyglet audio could not be initialized')


class PygletSound(object):
    """Use pyglet audio capabilities"""
    def __init__(self, ec, stim_fs):
        logger.info('Expyfun: Setting up Pyglet audio')
        self.fs = 44100

        # Need to generate at RMS=1 to match TDT circuit
        noise = np.random.normal(0, 1.0, int(self.fs * 15.0))  # 15 secs

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
        self.clear_buffer()  # initializes self.audio
        self.ec = ec
        self._noise_playing = True
        logger.debug('Expyfun: Pyglet sound server started')
        flush_logger()

    def start_noise(self):
        self.noise.play()
        self._noise_playing = True

    def stop_noise(self):
        self.noise.stop()

    def clear_buffer(self):
        self.audio = SoundPlayer(np.zeros((2, 1)), self.fs)

    def load_buffer(self, samples):
        self.audio = SoundPlayer(samples.T, self.fs)

    def play(self):
        self.audio.play()
        self.ec._stamp_ttl_triggers([1])

    def stop(self):
        self.audio.stop()

    def set_noise_level(self, level):
        new_noise = SoundPlayer(self.noise_array * level, self.fs, loop=True)
        if self._noise_playing:
            self.noise.stop()
            new_noise.play()
        self.noise = new_noise

    def halt(self):
        self.stop()
        self.stop_noise()
