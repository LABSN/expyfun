"""python-sounddevice interface for sound output."""

# Authors: Eric Larson <larsoner@uw.edu>
#          Thomas Stoll <tomstoll@med.umich.edu>
#
# License: BSD (3-clause)

import sys
import threading
from collections import deque

import numpy as np
import sounddevice as sd

from .._utils import get_config, logger

_PRIORITY = 300


class _StreamRegistry(dict):
    def __del__(self):
        for key, stream in list(self.items()):
            logger.debug(f"Closing {stream}")
            if getattr(stream, "closed", False):
                continue
            stream.abort()
            stream.close()
        self.clear()

    def _get_stream(self, fs, n_channels, api, name, api_options):
        """Select the API and device."""
        if api is None:
            api = get_config("SOUND_CARD_API", None)
        if api is None:
            api = dict(
                darwin="Core Audio",
                win32="Windows WASAPI",
                linux="ALSA",
                linux2="ALSA",
            )[sys.platform]
        key = (fs, n_channels, api, name)
        stream = self.get(key)
        if stream is not None and getattr(stream, "closed", False):
            # Remove closed streams from the registry
            del self[key]
            stream = None
        if stream is None:
            self[key] = _init_stream(fs, n_channels, api, name, api_options)
        return self[key]


_stream_registry = _StreamRegistry()


class _Stream(sd.OutputStream):
    """Output stream that plays queued buffers back to back (gapless).

    Looped buffers (e.g., noise) are mixed on top of the queue.
    """

    def __init__(self, **kwargs):
        self._queue = deque()  # entries are [player, data, position]
        self._loops = []  # same entry format
        self._lock = threading.Lock()
        self._underflows = 0
        self._frames = 0  # size of the most recent callback block
        super().__init__(callback=self._callback, **kwargs)

    def _callback(self, outdata, frames, time, status):
        outdata.fill(0)
        offset = 0
        with self._lock:
            self._underflows += bool(status.output_underflow)
            self._frames = frames
            while self._queue and offset < frames:
                entry = self._queue[0]
                n = min(frames - offset, len(entry[1]) - entry[2])
                outdata[offset : offset + n] = entry[1][entry[2] : entry[2] + n]
                offset += n
                entry[2] += n
                if entry[2] >= len(entry[1]):
                    self._queue.popleft()
            for entry in self._loops:
                idx = np.arange(entry[2], entry[2] + frames)
                outdata += np.take(entry[1], idx, axis=0, mode="wrap")
                entry[2] = (entry[2] + frames) % len(entry[1])


def _init_stream(fs, n_channels, api, name, api_options=None):
    device, param_str, extra_settings, all_devices = _find_device(
        n_channels, api, name, api_options
    )
    del api, name, api_options
    if fs is not None:
        param_str += " @ %d Hz" % (fs,)
    blocksize = int((fs or device["default_samplerate"]) / 5)
    try:
        stream = _Stream(
            samplerate=fs,
            blocksize=blocksize,
            device=device["index"],
            channels=n_channels,
            dtype="float32",
            latency="low",
            dither_off=True,
            extra_settings=extra_settings,
        )
    except Exception:
        raise RuntimeError(
            f"Could not set up {param_str}, all options:\n\n{all_devices}"
        )
    assert stream.channels == n_channels
    if fs is None:
        param_str += " @ %d Hz" % (stream.samplerate,)
    else:
        assert stream.samplerate == fs
    logger.info(
        "Expyfun: using %s, %0.1f ms nominal latency"
        % (param_str, 1000 * device["default_low_output_latency"])
    )
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
        self._loop = bool(loop)
        if fixed_delay is not None:
            if isinstance(fixed_delay, str):
                try:
                    fixed_delay = float(fixed_delay)
                except ValueError:
                    raise ValueError(
                        "fixed_delay must be None or 0, got %s" % fixed_delay
                    )
            if fixed_delay != 0:
                raise RuntimeError("fixed_delay must be 0 for gapless playback.")
        self._n_samples, n_channels = self._data.shape
        assert n_channels >= 1
        self._n_channels = n_channels
        self._stream = None  # in case the next line crashes, __del__ works
        self._stream = _stream_registry._get_stream(
            fs, self._n_channels, api, name, api_options
        )
        self._fs = float(self._stream.samplerate)
        self._ec_duration = self._n_samples / self._fs
        self._fixed_delay = fixed_delay

    @property
    def fs(self):
        return self._fs

    @property
    def playing(self):
        if self._stream is None:
            return False
        with self._stream._lock:
            return any(player is self for player, _, _ in self._stream._loops) or any(
                player is self and pos < len(data)
                for player, data, pos in self._stream._queue
            )

    @property
    def _pending(self):  # more than one block left to hand off (keeps gapless)
        with self._stream._lock:
            return any(
                player is self and len(data) - pos > self._stream._frames
                for player, data, pos in self._stream._queue
            )

    @property
    def _start_time(self):
        return 0.0

    def play(self):
        """Play (queued after any audio already playing on this stream)."""
        if self._stream is not None:
            with self._stream._lock:
                entries = self._stream._loops if self._loop else self._stream._queue
                entries.append([self, self._data, 0])
                underflows, self._stream._underflows = self._stream._underflows, 0
            if underflows:
                logger.exp(f"Expyfun: {underflows} sound card underflow(s)")
            if not self._stream.active:
                self._stream.start()

    def stop(self, wait=False, extra_delay=0.0):
        """Stop this player's audio after extra_delay sec (wait is ignored)."""
        if self._stream is not None:
            n_keep = int(round(extra_delay * self._fs))
            with self._stream._lock:
                self._stream._loops[:] = [
                    entry for entry in self._stream._loops if entry[0] is not self
                ]
                for entry in self._stream._queue:
                    if entry[0] is self:  # the callback drops finished entries
                        entry[1] = entry[1][: entry[2] + n_keep]

    def delete(self):
        if getattr(self, "_stream", None) is not None:
            self.stop()
            self._stream = None


def _abort_all_queues():
    for key, stream in list(_stream_registry.items()):
        if getattr(stream, "closed", False):
            continue
        stream.abort(ignore_errors=True)
        stream.close(ignore_errors=True)
        del _stream_registry[key]


def _find_device(n_channels, api, name, api_options=None):
    devices = sd.query_devices()
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
        index = api["default_output_device"]
        indices = [device["index"] for device in devices]
        name = devices[indices.index(index)]["name"]
        logger.exp(f"Selected default sound device: {name!r}")
    all_devices = []
    for d in devices:
        if d["max_output_channels"] < 1:
            continue
        all_pairs = " | ".join(
            f"{k}={v}"
            for k, v in d.items()
            if not (k.startswith("default_") or k == "index")
        )
        all_devices.append(f"{d['index']}: <{all_pairs}>")
    all_devices = "\n".join(all_devices)
    for device in devices:
        if (
            device["hostapi"] == ai
            and device["max_output_channels"] >= n_channels
            and name in device["name"]
        ):
            break
    else:
        raise RuntimeError(
            f"Could not find device on API {api['name']} with name "
            f"containing {name!r} with at least {n_channels=}. "
            f"Found:\n{all_devices}"
        )
    param_str = (
        f"sound card {device['name']!r} (index={device['index']}) via {api['name']}"
    )
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
    return device, param_str, extra_settings, all_devices
