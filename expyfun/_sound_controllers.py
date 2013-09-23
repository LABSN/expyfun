"""Hardware interfaces for sound output"""

# Authors: Dan McCloy <drmccloy@uw.edu>
#          Eric Larson <larsoner@uw.edu>
#
# License: BSD (3-clause)

import numpy as np
from scipy import fftpack
from psychopy import sound
from psychopy.constants import STARTED, STOPPED


class PsychSound(object):
    """Use PsychoPy audio capabilities"""
    def __init__(self, ec, stim_fs):
        if sound.Sound is None:
            raise ImportError('PsychoPy sound could not be initialized. '
                              'Ensure you have the pygame package properly'
                              ' installed.')
        self.fs = 44100
        self.audio = sound.Sound(np.zeros((1, 2)), sampleRate=self.fs)
        self.audio.setVolume(1.0, log=False)  # dont change: linearity unknown
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
        self.noise = sound.Sound(self.noise_array, sampleRate=self.fs)
        self.noise.setVolume(1.0, log=False)  # dont change: linearity unknown
        self.ec = ec

    def start_noise(self):
        self.noise.play(loops=-1)
        self.noise.status = STARTED

    def stop_noise(self):
        self.noise.stop()
        self.noise.status = STOPPED

    def clear_buffer(self):
        self.audio.setSound(np.zeros((1, 2)), log=False)

    def load_buffer(self, samples):
        self.audio = sound.Sound(samples, sampleRate=self.fs)
        self.audio.setVolume(1.0, log=False)  # dont change: linearity unknown

    def play(self):
        self.audio.play()
        self.ec.stamp_triggers([1])

    def stop(self):
        self.audio.stop()

    def set_noise_level(self, level):
        new_noise = sound.Sound(self.noise_array * level, sampleRate=self.fs)
        if self.noise.status == STARTED:
            # change the noise level immediately
            self.noise.stop()
            self.noise = new_noise
            self.noise._snd.play(loops=-1)
            self.noise.status = STARTED  # have to explicitly set status,
            # since we bypass PsychoPy's play() method to access pygame "loops"
        else:
            self.noise = new_noise

    def halt(self):
        sound.pyoSndServer.shutdown()
