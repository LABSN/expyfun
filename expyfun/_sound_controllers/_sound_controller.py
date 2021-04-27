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

_SOUND_CARD_KEYS = (
    'TYPE', 'SOUND_CARD_BACKEND', 'SOUND_CARD_API',
    'SOUND_CARD_NAME', 'SOUND_CARD_FS', 'SOUND_CARD_FIXED_DELAY',
    'SOUND_CARD_TRIGGER_CHANNELS', 'SOUND_CARD_API_OPTIONS',
    'SOUND_CARD_TRIGGER_SCALE', 'SOUND_CARD_TRIGGER_INSERTION',
    'SOUND_CARD_TRIGGER_ID_AFTER_ONSET',
)


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
    trigger_duration : float
        The duration (sec) to use for triggers (if applicable).
    ec : instance of ExperimentController
        The ExperimentController.

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
    - 'SOUND_CARD_TRIGGER_CHANNELS' : int
        Number of sound card channels to use as stim channels.
    - 'SOUND_CARD_API_OPTIONS': dict
        API options, such as ``{'exclusive': true}`` for WASAPI.
    - 'SOUND_CARD_TRIGGER_SCALE': float
        Scale factor for sound card triggers (after they are bit-shifted
        by 8). The default value (``1. / (2 ** 32 - 1)``) is meant to
        be appropriate for bit-perfect mapping to the 24 bit output of
        a SPDIF channel.
    - 'SOUND_CARD_TRIGGER_ID_AFTER_ONSET': bool
        If True, TTL IDs will be stored and stamped after the 1 trigger.

    Note that the defaults are superseded on individual machines by
    the configuration file.
    """

    def __init__(self, params, stim_fs, n_channels=2, trigger_duration=0.01,
                 ec=None):
        self.ec = ec
        defaults = dict(
            SOUND_CARD_BACKEND='auto',
            SOUND_CARD_TRIGGER_CHANNELS=0,
            SOUND_CARD_TRIGGER_SCALE=1. / float(2 ** 31 - 1),
            SOUND_CARD_TRIGGER_INSERTION='prepend',
            SOUND_CARD_TRIGGER_ID_AFTER_ONSET=False,
        )  # any omitted become None
        params = _check_params(params, _SOUND_CARD_KEYS, defaults, 'params')

        self.backend, self.backend_name = _import_backend(
            params['SOUND_CARD_BACKEND'])
        self._n_channels_stim = int(params['SOUND_CARD_TRIGGER_CHANNELS'])
        trig_scale = float(params['SOUND_CARD_TRIGGER_SCALE'])
        self._id_after_onset = (
            str(params['SOUND_CARD_TRIGGER_ID_AFTER_ONSET']).lower() == 'true')
        self._extra_onset_triggers = list()
        assert self._n_channels_stim >= 0
        self._n_channels = int(operator.index(n_channels))
        del n_channels
        insertion = str(params['SOUND_CARD_TRIGGER_INSERTION'])
        if insertion not in ('prepend', 'append'):
            raise ValueError('SOUND_CARD_TRIGGER_INSERTION must be "prepend" '
                             'or "append", got %r' % (insertion,))
        self._stim_sl = slice(None, None, 1 if insertion == 'prepend' else -1)
        extra = ''
        if self._n_channels_stim:
            extra = ('%d %sed stim and '
                     % (self._n_channels_stim, insertion))
        else:
            extra = ''
        del insertion
        logger.info('Expyfun: Setting up sound card using %s backend with %s'
                    '%d playback channels'
                    % (self.backend_name, extra, self._n_channels))
        self._kwargs = {key: params['SOUND_CARD_' + key.upper()] for key in (
            'fs', 'api', 'name', 'fixed_delay', 'api_options')}
        temp_sound = np.zeros((self._n_channels_tot, 1000))
        temp_sound = self.backend.SoundPlayer(temp_sound, **self._kwargs)
        self.fs = temp_sound.fs
        temp_sound.stop(wait=False)
        del temp_sound

        # Need to generate at RMS=1 to match TDT circuit, and use a power of
        # 2 length for the RingBuffer (here make it >= 15 sec)
        n_samples = 2 ** int(np.ceil(np.log2(self.fs * 15.)))
        noise = np.random.normal(0, 1.0, (self._n_channels, n_samples))

        # Low-pass if necessary
        if stim_fs < self.fs:
            # note we can use cheap DFT method here b/c
            # circular convolution won't matter for AWGN (yay!)
            freqs = rfftfreq(noise.shape[-1], 1. / self.fs)
            noise = rfft(noise, axis=-1)
            noise[:, np.abs(freqs) > stim_fs / 2.] = 0.0
            noise = irfft(noise, axis=-1)

        # ensure true RMS of 1.0 (DFT method also lowers RMS, compensate here)
        noise /= np.sqrt(np.mean(noise * noise))
        noise = np.concatenate(
            (np.zeros((self._n_channels_stim, noise.shape[1]), noise.dtype),
             noise))
        self.noise_array = noise
        self.noise_level = 0.01
        self.noise = None
        self.audio = None
        self.playing = False
        self._trigger_duration = trigger_duration
        self._trig_scale = trig_scale
        flush_logger()

    def __repr__(self):
        return ('<SoundController : %s playback %s trigger ch >'
                % (self._n_channels, self._n_channels_stim))

    @property
    def _n_channels_tot(self):
        return self._n_channels_stim + self._n_channels

    def start_noise(self):
        """Start noise."""
        if not self._noise_playing:
            self.noise = self.backend.SoundPlayer(
                self.noise_array * self.noise_level, loop=True, **self._kwargs)
            self.noise.play()

    def stop_noise(self, wait=False):
        """Stop noise.

        Parameters
        ----------
        wait : bool
            If True, wait for the action to complete.
            This is usually not necessary and can lead to tens of
            milliseconds of (variable) delay.
        """
        if self._noise_playing:
            self.noise.stop(wait=wait)
            self.noise.delete()
            self.noise = None

    @property
    def _noise_playing(self):
        return self.noise is not None

    def load_buffer(self, samples):
        """Load the buffer.

        Parameters
        ----------
        samples : ndarray, shape (n_samples, n_channels)
            The sound samples.
        """
        assert samples.ndim == 2
        self.stop(wait=False)
        if self.audio is not None:
            self.audio.delete()
            self.audio = None
        if self._n_channels_stim > 0:
            stim = self._make_digital_trigger([1] + self._extra_onset_triggers)
            extra = len(samples) - len(stim)
            if extra > 0:  # stim shorter than samples (typical)
                stim = np.pad(stim, ((0, extra), (0, 0)), 'constant')
            elif extra < 0:  # samples shorter than stim (very brief stim)
                samples = np.pad(samples, ((0, -extra), (0, 0)), 'constant')
            samples = np.concatenate((stim, samples)[self._stim_sl], axis=1)
        self.audio = self.backend.SoundPlayer(samples.T, **self._kwargs)

    def _make_digital_trigger(self, trigs, delay=None):
        if delay is None:
            delay = 2 * self._trigger_duration
        n_on = int(round(self.fs * self._trigger_duration))
        n_off = int(round(self.fs * (delay - self._trigger_duration)))
        n_each = n_on + n_off
        # At some point below we did:
        #
        #     (np.array(trigs, int) << 8) + 101
        #
        # The factor of 101 here could be, say, 127, since we bit shift by
        # 8. It should help ensure that we get the bit that we actually want
        # (plus only some low-order bits that will get discarded) in case
        # there is some rounding problem in however PortAudio and/or the OS
        # converts float32 to int32. In other words, there should be no
        # penalty/risk in up to (at least) 127 larger than the end int value
        # we want (e.g. we want 256 and get 256+127 after PA conversion),
        # but if we end up even 1 short (e.g., get 255 after conversion)
        # then we will lose the bit. Here we stay under 128 to avoid any
        # possible rounding error, though 255 in principle might even work.
        # HOWEVER -- if there is more than one trigger being sent at the same
        # time, this addition could be problematic. Hence we could add for
        # example 10 to give us some rounding-error buffer and be safe up
        # to 10 or so simultaneous triggers, but for now let's just see
        # if adding nothing is good enough (trust PortAudio to convert!).
        #
        # This is also why we keep our _trig_scale in float64, because it
        # can accurately represent a 32-bit integer without loss of precision,
        # whereas in 32-bit we get:
        #
        #     np.float32(2 ** 32 - 1) == np.float32(4294967295) == 4294967300
        #
        trigs = ((np.array(trigs, int) << 8) *
                 self._trig_scale).astype(np.float32)
        assert trigs.ndim == 1
        n_samples = n_each * len(trigs)
        stim = np.zeros((n_samples, self._n_channels_stim), np.float32)
        offset = 0
        for trig in trigs:
            stim[offset:offset + n_on] = trig
            offset += n_each
        return stim

    def stamp_triggers(self, triggers, delay=None, wait_for_last=True,
                       is_trial_id=False):
        """Stamp a list of triggers with a given inter-trigger delay.

        Parameters
        ----------
        triggers : list
            No input checking is done, so ensure triggers is a list,
            with each entry an integer with fewer than 8 bits (max 255).
        delay : float | None
            The inter-trigger-onset delay (includes "on" time).
            If None, will use twice the trigger duration (50% duty cycle).
        wait_for_last : bool
            If True, wait for last trigger to be stamped before returning.
        is_trial_id : bool
            If True and SOUND_CARD_TRIGGER_ID_AFTER_ONSET, the triggers will
            be stashed and appended to the 1 trigger for the sound onset.
        """
        if is_trial_id and self._id_after_onset:
            self._extra_onset_triggers = list(triggers)
            return
        if delay is None:
            delay = 2 * self._trigger_duration
        stim = self._make_digital_trigger(triggers, delay)
        stim = np.pad(
            stim, ((0, 0), (0, self._n_channels)[self._stim_sl]), 'constant')
        stim = self.backend.SoundPlayer(stim.T, **self._kwargs)
        stim.play()
        t_each = self._trigger_duration + delay
        duration = len(triggers) * t_each
        extra_delay = 0.1
        if wait_for_last:
            self.ec.wait_secs(duration)
        else:
            extra_delay += duration
        stim.stop(wait=False, extra_delay=extra_delay)

    def play(self):
        """Play."""
        assert not self.playing
        if self.audio is not None:
            self.audio.play()
        self.playing = True

    def stop(self, wait=False):
        """Stop.

        Parameters
        ----------
        wait : bool
            If True, wait for the action to complete.
            This is usually not necessary and can lead to tens of
            milliseconds of (variable) delay.
        """
        if self.audio is not None:
            self.audio.stop(wait=wait)
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
            self.stop_noise(wait=True)
            self.start_noise()
        else:
            self.noise = new_noise

    def halt(self):
        """Halt."""
        self.stop(wait=True)
        self.stop_noise(wait=True)


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
