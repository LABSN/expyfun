"""Hardware interfaces for sound output"""

# Authors: Dan McCloy <drmccloy@uw.edu>
#          Eric Larson <larsoner@uw.edu>
#
# License: BSD (3-clause)

import numpy as np
from scipy import fftpack
import sys

from ._utils import (HidePyoOutput, HideAlsaOutput, logger, wait_secs,
                     flush_logger, get_config)


class PyoSound(object):
    """Use Pyo audio capabilities"""
    def __init__(self, ec, stim_fs, buffer_size=128):
        # This is a hack for Travis, since it doesn't allow "audio" group on
        # linux, we need some way to fake audio calls
        logger.info('Expyfun: Setting up Pyo audio')
        self._pyo_server = None
        if get_config('_EXPYFUN_PYO_DUMMY_MODE', 'false') == 'true':
            # Use fake sound
            self.Sound = DummySound
            self.fs = stim_fs
        else:
            # nest the pyo import, in case we have a sys with just TDT
            self.Sound = Sound  # use real sound
            try:
                with HidePyoOutput():
                    import pyo
            except Exception as exp:
                raise RuntimeError('Cannot init pyo sound: {}'
                                   ''.format(str(exp)))

            driver_dict = dict(darwin='coreaudio', linux2='jack')
            audio = driver_dict.get(sys.platform, 'portaudio')
            self._pyo_server = pyo.Server(sr=44100, nchnls=2, audio=audio,
                                          buffersize=buffer_size)
            self._pyo_server.setVerbosity(1)  # error

            if sys.platform == 'win32':
                names, ids = pyo.pa_get_output_devices()
                driver, id_ = _best_driver_win32(names, ids)
                if not id_:
                    raise RuntimeError('No audio outputs found')
                logger.info('Using sound driver: {} (ID={})'
                            ''.format(driver, id_))
                self._pyo_server.setOutputDevice(id_)
            self._pyo_server.setDuplex(False)
            with HidePyoOutput():
                with HideAlsaOutput():
                    self._pyo_server.boot()
            if hasattr(self._pyo_server, 'getIsBooted'):
                if not self._pyo_server.getIsBooted():
                    raise RuntimeError('pyo sound server could not be booted')
            wait_secs(0.1)
            self._pyo_server.start()
            if not self._pyo_server.getIsStarted():
                raise RuntimeError('pyo sound server could not be started')
            self.fs = int(self._pyo_server.getSamplingRate())

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
        self.noise_array = np.array(np.c_[noise, -1.0 * noise], order='C')
        self.noise = self.Sound(self.noise_array, loop=True)
        self.clear_buffer()  # initializes self.audio
        self.ec = ec
        logger.debug('Expyfun: Pyo sound server started')
        flush_logger()

    def start_noise(self):
        self.noise.play()

    def stop_noise(self):
        self.noise.stop()

    def clear_buffer(self):
        self.audio = self.Sound(np.zeros((1, 2)))

    def load_buffer(self, samples):
        self.audio = self.Sound(samples)

    def play(self):
        self.audio.play()
        self.ec.stamp_triggers([1])

    def stop(self):
        self.audio.stop()

    def set_noise_level(self, level):
        new_noise = self.Sound(self.noise_array * level, loop=True)
        self.noise.stop()
        new_noise.play()
        self.noise = new_noise

    def halt(self):
        if self._pyo_server is not None:
            self._pyo_server.stop()
            self._pyo_server.shutdown()


def _best_driver_win32(devices, ids):
    """Find ASIO or Windows sound drivers"""
    prefs = ['ASIO', 'Primary Sound']  # 'Primary Sound' is DirectSound
    for pref in prefs:
        for driver, id_ in zip(devices, ids):
            if pref.encode('utf-8') in driver.encode('utf-8'):
                return driver, id_
    raise RuntimeError('could not find appropriate driver')


class Sound(object):
    """Create a sound object from numpy array"""
    def __init__(self, data, loop=False):
        import pyo
        _snd_table = pyo.DataTable(size=data.shape[0],
                                   init=data.T.tolist(),
                                   chnls=data.shape[1])
        self._snd = pyo.TableRead(_snd_table, freq=_snd_table.getRate(),
                                  loop=loop, mul=1.0)

    def play(self):
        """Starts playing the sound."""
        self._snd.out()

    def stop(self, log=True):
        """Stops the sound immediately"""
        self._snd.stop()


class DummySound(object):
    """Create a dummy sound object from numpy array"""
    def __init__(self, data, loop=False):
        pass

    def play(self):
        pass

    def stop(self, log=True):
        pass
