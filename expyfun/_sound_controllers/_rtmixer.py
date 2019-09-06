"""python-rtmixer interface for sound output."""

# Authors: Eric Larson <larsoner@uw.edu>
#
# License: BSD (3-clause)

import atexit
import sys

import numpy as np

from rtmixer import Mixer, RingBuffer
import sounddevice
from .._utils import logger, get_config

_PRIORITY = 100
_DEFAULT_NAME = None

# only initialize each mixer once and reuse it until Python closes
_MIXER_REGISTRY = {}


def _get_mixer(fs, n_channels, api, name, api_options):
    """Select the API and device."""
    # API
    if api is None:
        api = get_config('SOUND_CARD_API', None)
    if api is None:
        # Eventually we should maybe allow 'Windows WDM-KS',
        # 'Windows DirectSound', or 'MME'
        api = dict(
            darwin='Core Audio',
            win32='Windows WASAPI',
            linux='ALSA',
            linux2='ALSA',
        )[sys.platform]
    key = (fs, n_channels, api, name)
    if key not in _MIXER_REGISTRY:
        _MIXER_REGISTRY[key] = _init_mixer(fs, n_channels, api, name,
                                           api_options)
    return _MIXER_REGISTRY[key]


def _init_mixer(fs, n_channels, api, name, api_options=None):
    devices = sounddevice.query_devices()
    if len(devices) == 0:
        raise OSError('No sound devices found!')
    apis = sounddevice.query_hostapis()
    for ai, this_api in enumerate(apis):
        if this_api['name'] == api:
            api = this_api
            break
    else:
        raise RuntimeError('Could not find host API %s' % (api,))
    del this_api

    # Name
    if name is None:
        name = get_config('SOUND_CARD_NAME', None)
    if name is None:
        global _DEFAULT_NAME
        if _DEFAULT_NAME is None:
            di = api['default_output_device']
            _DEFAULT_NAME = devices[di]['name']
            logger.exp('Selected default sound device: %r' % (_DEFAULT_NAME,))
        name = _DEFAULT_NAME
    possible = list()
    for di, device in enumerate(devices):
        if device['hostapi'] == ai:
            possible.append(device['name'])
            if name in device['name']:
                break
    else:
        raise RuntimeError('Could not find device on API %r with name '
                           'containing %r, found:\n%s'
                           % (api['name'], name, '\n'.join(possible)))
    param_str = ('sound card %r (devices[%d]) via %r'
                 % (device['name'], di, api['name']))
    extra_settings = None
    if api_options is not None:
        if api['name'] == 'Windows WASAPI':
            # exclusive mode is needed for zero jitter on Windows in testing
            extra_settings = sounddevice.WasapiSettings(**api_options)
        else:
            raise ValueError(
                'api_options only supported for "Windows WASAPI" backend, '
                'using %s backend got api_options=%s'
                % (api['name'], api_options))
        param_str += ' with options %s' % (api_options,)
    param_str += ', %d channels' % (n_channels,)
    if fs is not None:
        param_str += ' @ %d Hz' % (fs,)
    try:
        mixer = Mixer(
            samplerate=fs, latency='low', channels=n_channels,
            dither_off=True, device=di,
            extra_settings=extra_settings)
    except Exception as exp:
        raise RuntimeError('Could not set up %s:\n%s' % (param_str, exp))
    assert mixer.channels == n_channels
    if fs is None:
        param_str += ' @ %d Hz' % (mixer.samplerate,)
    else:
        assert mixer.samplerate == fs

    mixer.start()
    try:
        mixer.start_time = mixer.time
    except Exception:
        mixer.start_time = 0
    logger.info('Expyfun: using %s, %0.1f ms nominal latency'
                % (param_str, 1000 * device['default_low_output_latency']))
    atexit.register(lambda: (mixer.abort(), mixer.close()))
    return mixer


class SoundPlayer(object):
    """SoundPlayer based on rtmixer."""

    def __init__(self, data, fs=None, loop=False, api=None, name=None,
                 fixed_delay=None, api_options=None):
        self._data = np.ascontiguousarray(
            np.clip(np.atleast_2d(data).T, -1, 1).astype(np.float32))
        self.loop = bool(loop)
        self._n_samples, n_channels = self._data.shape
        assert n_channels >= 1
        self._n_channels = n_channels
        self._mixer = None  # in case the next line crashes, __del__ works
        self._mixer = _get_mixer(fs, self._n_channels, api, name, api_options)
        if loop:
            self._ring = RingBuffer(self._data.itemsize * self._n_channels,
                                    self._data.size)
            self._ring.write(self._data)
        self._fs = float(self._mixer.samplerate)
        self._ec_duration = self._n_samples / self._fs
        self._action = None
        self._fixed_delay = fixed_delay
        if fixed_delay is not None:
            logger.info('Expyfun: Using fixed audio delay %0.1f ms'
                        % (1000 * fixed_delay,))
        else:
            logger.info('Expyfun: Variable audio delay')

    @property
    def fs(self):
        return self._fs

    @property
    def playing(self):
        return self._action is not None and self._mixer is not None

    def play(self):
        """Play."""
        if not self.playing and self._mixer is not None:
            if self._fixed_delay is not None:
                start = self._mixer.time + self._fixed_delay
            else:
                start = 0
            if self.loop:
                self._action = self._mixer.play_ringbuffer(
                    self._ring, start=start)
            else:
                self._action = self._mixer.play_buffer(
                    self._data, self._data.shape[1], start=start)

    def pause(self):
        """Pause."""
        if self.playing:
            action, self._action = self._action, None
            cancel_action = self._mixer.cancel(action)
            self._mixer.wait(cancel_action)

    def stop(self):
        """Stop."""
        self.pause()

    def delete(self):
        """Delete."""
        if getattr(self, '_mixer', None) is not None:
            self.pause()
            mixer, self._mixer = self._mixer, None
            stats = mixer.fetch_and_reset_stats().stats
            logger.exp('%d underflows %d blocks'
                       % (stats.output_underflows, stats.blocks))

    def __del__(self):  # noqa
        self.delete()
