"""Sound card interface for expyfun."""

# Authors: Dan McCloy <drmccloy@uw.edu>
#          Eric Larson <larsoner@uw.edu>
#
# License: BSD (3-clause)

import importlib
import operator
import os
import os.path as op

import numpy as np

from .._fixes import rfft, irfft, rfftfreq
from .._utils import logger, flush_logger, _check_params


_BACKENDS = tuple(sorted(
    op.splitext(x.lstrip('._'))[0] for x in os.listdir(op.dirname(__file__))
    if x.startswith('_') and x.endswith(('.py', '.pyc')) and
    not x.startswith(('_sound_controller.py', '__init__.py'))))
# libsoundio stub (kind of iffy)
# https://gist.github.com/larsoner/fd9228f321d369c8a00c66a246fcc83f


class SoundCardController(object):
    """Use a sound card.

    Parameters
    ----------
    params : dict
        A dictionary containing parameter keys. See Notes for details.
    stim_fs : float
        Stim fs, used to downsample the white noise if necessary.
    n_channels : int
        The number of playback channels to use.

    Notes
    -----
    Params should contain string values:

    - 'SOUND_CARD_BACKEND' : str
        The backend to use. Can be 'auto' (default), 'rtmixer', 'pyglet'.
    - 'SOUND_CARD_API' : str
        The API to use for the sound card.
        See :func:`sounddevice.query_hostapis`.
        The default is OS-dependent.
    - 'SOUND_CARD_NAME' : str
        The name as it is given by :func:`sounddevice.query_devices`.
        The default chooses the default sound card for the OS.
    - 'SOUND_CARD_FS' : float
        The sample rate to use for the sound card. The default
        lets the OS choose.
    - 'SOUND_CARD_FIXED_DELAY' : float
        The fixed delay (in sec) to use for playback.
        This is used by the rtmixer backend to ensure fixed
        latency playback.

    Note that the defaults are superseded on individual machines by
    the configuration file.
    """

    def __init__(self, params, stim_fs, n_channels=2):
        keys = ('TYPE', 'SOUND_CARD_BACKEND', 'SOUND_CARD_API',
                'SOUND_CARD_NAME', 'SOUND_CARD_FS', 'SOUND_CARD_FIXED_DELAY')
        defaults = dict(SOUND_CARD_BACKEND='auto')
        params = _check_params(params, keys, defaults, 'params')

        self.backend, self.backend_name = _import_backend(
            params['SOUND_CARD_BACKEND'])
        self._n_channels = operator.index(n_channels)
        logger.info('Expyfun: Setting up sound card audio using %s backend '
                    'and %d channels' % (self.backend_name, n_channels))
        self._kwargs = dict(
            fs=params.get('SOUND_CARD_FS', None),
            api=params.get('SOUND_CARD_API', None),
            name=params.get('SOUND_CARD_NAME', None),
            fixed_delay=params.get('SOUND_CARD_FIXED_DELAY', None),
        )
        temp_sound = np.zeros((self._n_channels, 1000))
        temp_sound = self.backend.SoundPlayer(temp_sound, **self._kwargs)
        self.fs = temp_sound.fs
        temp_sound.stop()
        del temp_sound

        # Need to generate at RMS=1 to match TDT circuit, and use a power of
        # 2 length for the RingBuffer (here make it >= 15 sec)
        n_samples = 2 ** int(np.ceil(np.log2(self.fs * 15.)))
        noise = np.random.normal(0, 1.0, (n_channels, n_samples))

        # Low-pass if necessary
        if stim_fs < self.fs:
            # note we can use cheap DFT method here b/c
            # circular convolution won't matter for AWGN (yay!)
            freqs = rfftfreq(noise.shape[-1], 1. / self.fs)
            noise = rfft(noise, axis=-1)
            noise[:, np.abs(freqs) > stim_fs / 2.] = 0.0
            noise = irfft(noise, axis=-1)

        # ensure true RMS of 1.0 (DFT method also lowers RMS, compensate here)
        self.noise_array = noise / np.sqrt(np.mean(noise * noise))
        self.noise_level = 0.01
        self.noise = None
        self.audio = None
        self.playing = False
        flush_logger()

    def start_noise(self):
        """Start noise."""
        if not self._noise_playing:
            self.noise = self.backend.SoundPlayer(
                self.noise_array * self.noise_level, loop=True, **self._kwargs)
            self.noise.play()

    def stop_noise(self):
        """Stop noise."""
        if self._noise_playing:
            self.noise.stop()
            self.noise.delete()
            self.noise = None

    @property
    def _noise_playing(self):
        return self.noise is not None

    def load_buffer(self, samples):
        """Load the buffer.

        Parameters
        ----------
        samples : ndarray
            The sound samples.
        """
        self.stop()
        if self.audio is not None:
            self.audio.delete()
            self.audio = None
        self.audio = self.backend.SoundPlayer(samples.T, **self._kwargs)

    def play(self):
        """Play."""
        assert not self.playing
        if self.audio is not None:
            self.audio.play()
        self.playing = True

    def stop(self):
        """Stop."""
        if self.audio is not None:
            self.audio.stop()
        self.playing = False

    def set_noise_level(self, level):
        """Set the noise level.

        Parameters
        ----------
        level : float
            The new level.
        """
        self.noise_level = float(level)
        new_noise = None
        if self._noise_playing:
            self.stop_noise()
            self.start_noise()
        else:
            self.noise = new_noise

    def halt(self):
        """Halt."""
        self.stop()
        self.stop_noise()


def _import_backend(backend):
    # Auto mode is special, will loop through all possible backends
    if backend == 'auto':
        backends = list()
        for backend in _BACKENDS:
            try:
                backends.append(_import_backend(backend)[0])
            except Exception:
                pass
        backends = sorted([backend._PRIORITY, backend] for backend in backends)
        if len(backends) == 0:
            raise RuntimeError('Could not load any sound backend: %s'
                               % (_BACKENDS,))
        backend = op.splitext(op.basename(backends[0][1].__file__))[0][1:]
    if backend not in _BACKENDS:
        raise ValueError('Unknown sound card backend %r, must be one of %s'
                         % (backend, ('auto',) + _BACKENDS))
    lib = importlib.import_module('._' + backend,
                                  package='expyfun._sound_controllers')
    return lib, backend


class SoundPlayer(object):
    """Play sounds via the sound card."""

    def __new__(self, data, **kwargs):
        """Create a new instance."""
        backend = kwargs.pop('backend', 'auto')
        return _import_backend(backend)[0].SoundPlayer(data, **kwargs)
