"""python-rtmixer interface for sound output."""

# Authors: Eric Larson <larsoner@uw.edu>
#
# License: BSD (3-clause)

import sys

import numpy as np
import sounddevice as sd
import warnings

from .._utils import get_config, logger

_PRIORITY = 99
_DEFAULT_NAME = None


class _StreamRegistry(dict):
    def __del__(self):
        for stream in self.values():
            print(f"Closing {stream}")
            stream.abort()
            stream.close()
        self.clear()

    def _get_stream(self, fs, n_channels, api, name, api_options):
        """Select the API and device."""
        # API
        if api is None:
            api = get_config("SOUND_CARD_API", None)
        if api is None:
            # Eventually we should maybe allow 'Windows WDM-KS',
            # 'Windows DirectSound', or 'MME'
            api = dict(
                darwin="Core Audio",
                win32="Windows WASAPI",
                linux="ALSA",
                linux2="ALSA",
            )[sys.platform]
        key = (fs, n_channels, api, name)
        if key not in self:
            self[key] = _init_stream(fs, n_channels, api, name, api_options)
        return self[key]


_stream_registry = _StreamRegistry()


def _init_stream(fs, n_channels, api, name, api_options=None):
    devices = sd.query_devices()
    if not isinstance(fs, int):
        try:
            fs = int(fs)
        except ValueError:
            raise RuntimeError("Sample rate must be an int.")
    if len(devices) == 0:
        raise OSError("No sound devices found!")
    apis = sd.query_hostapis()
    valid_apis = []
    for ai, this_api in enumerate(apis):
        if this_api["name"] == api:
            api = this_api
            break
        else:
            valid_apis.append(this_api["name"])
    else:
        m = 'Could not find host API %s. Valid choices are "%s"'
        raise RuntimeError(m % (api, ", ".join(valid_apis)))
    del this_api

    # Name
    if name is None:
        name = get_config("SOUND_CARD_NAME", None)
    if name is None:
        global _DEFAULT_NAME
        if _DEFAULT_NAME is None:
            di = api["default_output_device"]
            _DEFAULT_NAME = devices[di]["name"]
            logger.exp("Selected default sound device: %r" % (_DEFAULT_NAME,))
        name = _DEFAULT_NAME
    possible = list()
    for di, device in enumerate(devices):
        if device["hostapi"] == ai:
            possible.append(device["name"])
            if name in device["name"]:
                break
    else:
        raise RuntimeError(
            "Could not find device on API %r with name "
            "containing %r, found:\n%s" % (api["name"], name, "\n".join(possible))
        )
    param_str = "sound card %r (devices[%d]) via %r" % (device["name"], di, api["name"])
    extra_settings = None
    if api_options is not None:
        if api["name"] == "Windows WASAPI":
            # exclusive mode is needed for zero jitter on Windows in testing
            extra_settings = sd.WasapiSettings(**api_options)
        else:
            raise ValueError(
                'api_options only supported for "Windows WASAPI" backend, '
                "using %s backend got api_options=%s" % (api["name"], api_options)
            )
        param_str += " with options %s" % (api_options,)
    param_str += ", %d channels" % (n_channels,)
    if fs is not None:
        param_str += " @ %d Hz" % (fs,)
    try:
        stream = sd.OutputStream(samplerate=fs, blocksize=fs//5, device=di,
                                 channels=n_channels, dtype="float32",
                                 latency='low', dither_off=True,
                                 extra_settings=extra_settings)
    except Exception as exp:
        raise RuntimeError(f"Could not set up {param_str}:\n{exp}") from None
    assert stream.channels == n_channels
    if fs is None:
        param_str += " @ %d Hz" % (stream.samplerate,)
    else:
        assert stream.samplerate == fs

    return stream


class SoundPlayer:
    """SoundPlayer based on sounddevice."""

    def __init__(
        self,
        data,
        fs=None,
        loop=False,
        api=None,
        name=None,
        fixed_delay=None,
        api_options=None,
    ):
        data = np.atleast_2d(data).T
        data = np.asarray(data, np.float32, "C")
        self._data = data
        if loop is not False or fixed_delay is not None:
            raise NotImplementedError("Not implemented for sounddevice backend.")
        self._n_samples, n_channels = self._data.shape
        assert n_channels >= 1
        self._n_channels = n_channels
        self._stream = None  # in case the next line crashes, __del__ works
        self._stream = _stream_registry._get_stream(
            fs, self._n_channels, api, name, api_options
        )
        self._fs = float(self._stream.samplerate)
        self._ec_duration = self._n_samples / self._fs
        self._action = None
        self._fixed_delay = fixed_delay

    @property
    def fs(self):
        return self._fs

    @property
    def playing(self):
        return self._action is not None and self._stream is not None

    @property
    def _start_time(self):
        return 0.0

    def play(self):
        """Play."""
        if not self.playing:
            self._stream.start()
        if self._stream is not None:
            self._stream.write(self._data)

    def stop(self, wait=False):
        """Stop. - Currently does nothing."""
        if self.playing:
            self._stream.abort()

    def delete(self):
        if getattr(self, "_stream", None) is not None:
            self.stop()
            stream, self._stream = self._stream, None


def _abort_all_queues():
    for stream in _stream_registry.values():
        do_start_stop = stream.stopped
        if do_start_stop:
            stream.start()
        if do_start_stop:
            stream.abort(ignore_errors=False)
