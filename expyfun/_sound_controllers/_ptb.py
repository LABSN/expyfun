"""Psychtoolbox-3 interface for sound output."""

# Authors: Eric Larson <larsoner@uw.edu>
#
# License: BSD (3-clause)

import sys
import numpy as np

import psychtoolbox
from psychtoolbox import audio

from .._utils import logger, get_config

_DEFAULT_NAME = None
_PRIORITY = 200

# only initialize each mixer once and reuse it until Python closes
_STREAM_REGISTRY = {}


def _get_stream(fs, n_channels, api, name, api_options):
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
    key = (fs, n_channels, api, name, api_options)
    if key not in _STREAM_REGISTRY:
        _STREAM_REGISTRY[key] = _init_stream(
            fs, n_channels, api, name, api_options)
    return _STREAM_REGISTRY[key]


def _init_stream(fs, n_channels, api, name, api_options=None):
    logger.info('Using PsychPortAudio version %s'
                % (audio.get_version_info()['version'],))
    audio.verbosity(5)  # XXX debug
    device_type = {
        'ALSA': 8,
        'Core Audio': 5,
        'Windows WASAPI': 13,
    }[api]
    del api
    devices = audio.get_devices(device_type)
    device_names = [d['DeviceName'].strip().replace('\n', ' ')
                    for d in devices]
    if len(devices) == 0:
        raise OSError('No sound devices found!')

    # Name
    if name is None:
        name = get_config('SOUND_CARD_NAME', None)
    if name is None:
        global _DEFAULT_NAME
        if _DEFAULT_NAME is None:
            di = 0
            _DEFAULT_NAME = devices[di]['DeviceName']
            logger.exp('Selected default sound device: %r' % (_DEFAULT_NAME,))
        name = _DEFAULT_NAME
    for di, device in enumerate(device_names):
        if name in device:
            break
    else:
        raise RuntimeError('Could not find device with name '
                           'containing %r, found:\n%s'
                           % (name, '\n'.join(devices)))
    device = devices[di]
    del device_names
    param_str = ('sound card %r (devices[%d])' % (device['DeviceName'], di))
    param_str += ', %d channels' % (n_channels,)
    try:
        stream = audio.Stream(
            device_id=device['DeviceIndex'],
            mode=9,  # master 8 + playback 1
            latency_class=4, freq=fs, channels=n_channels)
        stream.start(0, 0, 1)
    except Exception as exp:
        raise RuntimeError('Could not set up %s:\n%s' % (param_str, exp))
    stream.fs = stream.status['SampleRate']
    if fs is not None:
        assert stream.fs == fs
    param_str += ' @ %d Hz' % (fs,)
    logger.info('Expyfun: using %s, %0.1f ms nominal latency'
                % (param_str, 1000 * device['LowOutputLatency']))
    return stream


class SoundPlayer(object):
    """SoundPlayer based on Psychtoolbox-3."""

    def __init__(self, data, fs=None, loop=False, api=None, name=None,
                 fixed_delay=None, api_options=None):
        self.playing = False
        self._data = np.ascontiguousarray(
            np.clip(np.atleast_2d(data).T, -1, 1).astype(np.float32))
        self.loop = bool(loop)
        self._n_samples, n_channels = self._data.shape
        assert n_channels >= 1
        self._n_channels = n_channels
        self._stream = _get_stream(
            fs, self._n_channels, api, name, api_options)
        self._loops = int(not loop)  # 0 means repeat, otherwise play one

        self._fs = float(self._stream.fs)
        self._ec_duration = self._n_samples / self._fs
        self._track = audio.Slave(self._stream.handle, data=self._data)
        self._fixed_delay = fixed_delay
        if fixed_delay is not None:
            logger.info('Expyfun: Using fixed audio delay %0.1f ms'
                        % (1000 * fixed_delay,))
        else:
            logger.info('Expyfun: Variable audio delay')

    @property
    def fs(self):
        return self._fs

    def play(self):
        """Play."""
        if not self.playing and self._stream is not None:
            if self._fixed_delay is not None:
                start = psychtoolbox.GetSecs() + self._fixed_delay
            else:
                start = 0
            self._track.start(repetitions=self._loops, when=start)
            self.playing = True

    def pause(self):
        """Pause."""
        if self.playing:
            self._track.stop(block_until_stopped=0)
            self.playing = False

    def stop(self):
        """Stop."""
        self.pause()

    def delete(self):
        """Delete."""
        if getattr(self, '_stream', None) is not None:
            self.pause()
            self._track.stop(block_until_stopped=0)
            self._track.close()
            self._track = self._stream = None

    def __del__(self):  # noqa
        self.delete()
