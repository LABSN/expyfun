"""python-rtmixer interface for sound output."""

# Authors: Eric Larson <larsoner@uw.edu>
#
# License: BSD (3-clause)

import sys

import numpy as np
from rtmixer import Mixer, RingBuffer

from .._utils import get_config, logger
from ._sounddevice import _find_device

_PRIORITY = 100


# only initialize each mixer once and reuse it until this gets garbage
# collected


class _MixerRegistry(dict):
    def __del__(self):
        for mixer in self.values():
            logger.debug(f"Closing {mixer}")
            mixer.abort()
            mixer.close()
        self.clear()

    def _get_mixer(self, fs, n_channels, api, name, api_options):
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
            self[key] = _init_mixer(fs, n_channels, api, name, api_options)
        return self[key]


_mixer_registry = _MixerRegistry()


def _init_mixer(fs, n_channels, api, name, api_options=None):
    device, param_str, extra_settings, all_devices = _find_device(
        n_channels,
        api,
        name,
        api_options=api_options,
    )
    if fs is not None:
        param_str += " @ %d Hz" % (fs,)
    try:
        mixer = Mixer(
            samplerate=fs,
            latency="low",
            channels=n_channels,
            dither_off=True,
            device=device["index"],
            extra_settings=extra_settings,
        )
    except Exception:
        raise RuntimeError(f"Could not set up {param_str}:\n\n{all_devices}")
    assert mixer.channels == n_channels
    if fs is None:
        param_str += " @ %d Hz" % (mixer.samplerate,)
    else:
        assert mixer.samplerate == fs

    mixer.start()
    assert mixer.active
    logger.info(
        "Expyfun: using %s, %0.1f ms nominal latency"
        % (param_str, 1000 * device["default_low_output_latency"])
    )
    return mixer


class SoundPlayer:
    """SoundPlayer based on rtmixer."""

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
        self.loop = bool(loop)
        self._n_samples, n_channels = self._data.shape
        assert n_channels >= 1
        self._n_channels = n_channels
        self._mixer = None  # in case the next line crashes, __del__ works
        self._mixer = _mixer_registry._get_mixer(
            fs, self._n_channels, api, name, api_options
        )
        if loop:
            self._ring = RingBuffer(
                self._data.itemsize * self._n_channels, self._data.size
            )
            self._ring.write(self._data)
        self._fs = float(self._mixer.samplerate)
        self._ec_duration = self._n_samples / self._fs
        self._action = None
        self._fixed_delay = fixed_delay
        if fixed_delay is not None:
            logger.info(
                "Expyfun: Using fixed audio delay %0.1f ms" % (1000 * fixed_delay,)
            )
        else:
            logger.info("Expyfun: Variable audio delay")

    @property
    def fs(self):
        return self._fs

    @property
    def playing(self):
        return self._action is not None and self._mixer is not None

    @property
    def _start_time(self):
        if self._fixed_delay is not None:
            return self._mixer.time + self._fixed_delay
        else:
            return 0.0

    def play(self):
        """Play."""
        if not self.playing and self._mixer is not None:
            if self.loop:
                self._action = self._mixer.play_ringbuffer(
                    self._ring, start=self._start_time
                )
            else:
                self._action = self._mixer.play_buffer(
                    self._data, self._data.shape[1], start=self._start_time
                )

    def stop(self, wait=True, extra_delay=0.0):
        """Stop."""
        if self.playing:
            action, self._action = self._action, None
            # Impose the same delay here that we imposed on the stim start
            cancel_action = self._mixer.cancel(
                action, time=self._start_time + extra_delay
            )
            if wait:
                self._mixer.wait(cancel_action)
            else:
                return cancel_action

    def delete(self):
        """Delete."""
        if getattr(self, "_mixer", None) is not None:
            self.stop(wait=False)
            mixer, self._mixer = self._mixer, None
            try:
                stats = mixer.fetch_and_reset_stats().stats
            except RuntimeError as exc:  # action queue is full
                logger.exp(f"Could not fetch mixer stats ({exc})")
            else:
                logger.exp(
                    f"{stats.output_underflows} underflows {stats.blocks} blocks"
                )

    def __del__(self):  # noqa
        self.delete()


def _abort_all_queues():
    for mixer in _mixer_registry.values():
        if len(mixer.actions) == 0:
            continue
        do_start_stop = mixer.stopped
        if do_start_stop:
            mixer.start()
        for action in list(mixer.actions):
            mixer.wait(mixer.cancel(action))
        mixer.wait()
        assert len(mixer.actions) == 0, mixer.actions
        if do_start_stop:
            mixer.abort(ignore_errors=False)
        assert len(mixer.actions) == 0, mixer.actions
