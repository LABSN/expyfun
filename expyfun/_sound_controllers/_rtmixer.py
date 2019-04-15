"""python-rtmixer interface for sound output."""

# Authors: Eric Larson <larsoner@uw.edu>
#
# License: BSD (3-clause)

import sys

import numpy as np

from rtmixer import Mixer, RingBuffer
from sounddevice import query_devices, query_hostapis
from .._utils import logger, get_config

_PRIORITY = 100
_DEFAULT_NAME = None


# XXX we should really use the sys config to choose the fs, then resample
# in software...

def _init_mixer(fs, n_channels, api=None, name=None):
    """Select the API and device."""
    devices = query_devices()
    if len(devices) == 0:
        raise OSError('No sound devices found!')
    apis = query_hostapis()
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
        )[sys.platform]
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
            print('Selected default sound device: %r' % (_DEFAULT_NAME,))
        name = _DEFAULT_NAME
    possible = list()
    for di, device in enumerate(devices):
        if device['hostapi'] == ai:
            possible.append(device['name'])
            if name == device['name']:
                break
    else:
        raise RuntimeError('Could not find device on API %r with name '
                           'containing %r, found:\n%s'
                           % (api['name'], name, '\n'.join(possible)))
    param_str = ('sound card %r (devices[%d]) via %r, %d channels'
                 % (device['name'], di, api['name'], n_channels))
    if fs is not None:
        param_str += ' @ %d Hz' % (fs,)
    try:
        mixer = Mixer(
            samplerate=fs, latency='low', channels=n_channels,
            dither_off=True, device=di)
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
    logger.info('Expyfun: using %s, %0.1f ms latency'
                % (param_str, 1000 * device['default_low_output_latency']))
    return mixer


class SoundPlayer(object):
    """SoundPlayer based on rtmixer."""

    def __init__(self, data, fs=None, loop=False, api=None, name=None):
        self._data = np.ascontiguousarray(
            np.clip(data.T, -1, 1).astype(np.float32))
        self.loop = bool(loop)
        self._n_samples, n_channels = self._data.shape
        assert n_channels in (1, 2)
        if n_channels == 1:
            self._data = np.tile(self._data, (2, 1))
            n_channels = 2
        self._n_channels = n_channels
        self._mixer = None  # in case the next line crashes, __del__ works
        self._mixer = _init_mixer(fs, self._n_channels, api, name)
        if loop:
            self._ring = RingBuffer(self._data.itemsize * self._n_channels,
                                    self._data.size)
            self._ring.write(self._data)
        self._fs = float(self._mixer.samplerate)
        self._ec_duration = self._n_samples / self._fs
        self._action = None

    @property
    def fs(self):
        return self._fs

    @property
    def playing(self):
        return self._action is not None and self._mixer is not None

    def play(self):
        """Play."""
        if not self.playing:
            if self.loop:
                self._action = self._mixer.play_ringbuffer(self._ring)
            else:
                self._action = self._mixer.play_buffer(
                    self._data, self._data.shape[1])
                #    start=self._mixer.time - self._mixer.start_time + 0.06)

    def pause(self):
        """Pause."""
        if self.playing:
            action, self._action = self._action, None
            self._mixer.cancel(action)

    def stop(self):
        """Stop."""
        self.pause()

    def delete(self):
        """Delete."""
        if self._mixer is not None:
            self.pause()
            mixer, self._mixer = self._mixer, None
            stats = mixer.fetch_and_reset_stats().stats
            print('%d underflows %d blocks'
                  % (stats.output_underflows, stats.blocks))
            mixer.abort()
            mixer.close()

    def __del__(self):
        self.delete()
