"""Sound card interface for expyfun."""

# Authors: Dan McCloy <drmccloy@uw.edu>
#          Eric Larson <larsoner@uw.edu>
#
# License: BSD (3-clause)

import importlib
import operator
import os
import os.path as op
import warnings

import numpy as np

from .._fixes import irfft, rfft, rfftfreq
from .._utils import _check_params, _fix_audio_dims, flush_logger, logger

_BACKENDS = tuple(
    sorted(
        op.splitext(x.lstrip("._"))[0]
        for x in os.listdir(op.dirname(__file__))
        if x.startswith("_")
        and x.endswith((".py", ".pyc"))
        and not x.startswith(("_sound_controller.py", "__init__.py"))
    )
)
# libsoundio stub (kind of iffy)
# https://gist.github.com/larsoner/fd9228f321d369c8a00c66a246fcc83f

_SOUND_CARD_KEYS = (
    "TYPE",
    "SOUND_CARD_BACKEND",
    "SOUND_CARD_API",
    "SOUND_CARD_NAME",
    "SOUND_CARD_FS",
    "SOUND_CARD_FIXED_DELAY",
    "SOUND_CARD_TRIGGER_CHANNELS",
    "SOUND_CARD_API_OPTIONS",
    "SOUND_CARD_TRIGGER_SCALE",
    "SOUND_CARD_TRIGGER_INSERTION",
    "SOUND_CARD_TRIGGER_ID_AFTER_ONSET",
    "SOUND_CARD_DRIFT_TRIGGER",
)


class SoundCardController:
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
        The backend to use. Can be 'auto' (default), 'rtmixer', 'pyglet', 'sounddevice'.
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
    - 'SOUND_CARD_DRIFT_TRIGGER': list-like
        Defaults to ['end'] which places a 2 trigger at the very end of the
        trial. Can also be a scalar or list of scalars to insert 2
        triggers at the time of the scalar(s) (in sec). Negative values will be
        interpreted as time from end of trial.

    Note that the defaults are superseded on individual machines by
    the configuration file.
    """

    def __init__(self, params, stim_fs, n_channels=2, trigger_duration=0.01, ec=None):
        self.ec = ec
        defaults = dict(
            SOUND_CARD_BACKEND="auto",
            SOUND_CARD_TRIGGER_CHANNELS=0,
            SOUND_CARD_TRIGGER_SCALE=1.0 / float(2**31 - 1),
            SOUND_CARD_TRIGGER_INSERTION="prepend",
            SOUND_CARD_TRIGGER_ID_AFTER_ONSET=False,
            SOUND_CARD_DRIFT_TRIGGER="end",
        )  # any omitted become None
        params = _check_params(params, _SOUND_CARD_KEYS, defaults, "params")
        if params["SOUND_CARD_FS"] is not None:
            params["SOUND_CARD_FS"] = float(params["SOUND_CARD_FS"])

        self.backend, self.backend_name = _import_backend(params["SOUND_CARD_BACKEND"])
        self._n_channels_stim = int(params["SOUND_CARD_TRIGGER_CHANNELS"])
        trig_scale = float(params["SOUND_CARD_TRIGGER_SCALE"])
        self._id_after_onset = (
            str(params["SOUND_CARD_TRIGGER_ID_AFTER_ONSET"]).lower() == "true"
        )
        self._extra_onset_triggers = list()
        drift_trigger = params["SOUND_CARD_DRIFT_TRIGGER"]
        if np.isscalar(drift_trigger):
            drift_trigger = [drift_trigger]
        # convert possible command-line option
        if isinstance(drift_trigger, str) and drift_trigger != "end":
            drift_trigger = eval(drift_trigger)
        if isinstance(drift_trigger, str):
            drift_trigger = [drift_trigger]
        assert isinstance(drift_trigger, (list, tuple)), type(drift_trigger)
        drift_trigger = list(drift_trigger)  # make mutable
        for trig in drift_trigger:
            if isinstance(trig, str):
                assert trig == "end", trig
            else:
                assert isinstance(trig, (int, float)), type(trig)
        self._drift_trigger_time = drift_trigger
        assert self._n_channels_stim >= 0
        self._n_channels = int(operator.index(n_channels))
        del n_channels
        insertion = str(params["SOUND_CARD_TRIGGER_INSERTION"])
        if insertion not in ("prepend", "append"):
            raise ValueError(
                'SOUND_CARD_TRIGGER_INSERTION must be "prepend" '
                'or "append", got %r' % (insertion,)
            )
        self._stim_sl = slice(None, None, 1 if insertion == "prepend" else -1)
        extra = ""
        if self._n_channels_stim:
            extra = "%d %sed stim and " % (self._n_channels_stim, insertion)
        else:
            extra = ""
        del insertion
        logger.info(
            "Expyfun: Setting up sound card using %s backend with %s"
            "%d playback channels" % (self.backend_name, extra, self._n_channels)
        )
        if self.backend_name == "pyglet":
            # pyglet doesn't allow specifying api, name, fixed_delay, or api_options
            self._kwargs = dict(fs=params["SOUND_CARD_FS"])
        else:
            self._kwargs = {
                key: params["SOUND_CARD_" + key.upper()]
                for key in ("fs", "api", "name", "fixed_delay", "api_options")
            }
        if self.backend_name == "sounddevice":  # use this or next line
            # ensure id triggers are after onset for gapless playback
            if not self.ec._gapless:
                raise NotImplementedError(
                    "Currently, only gapless=True is allowed for sounddevice backend"
                )
            else:
                if not self._id_after_onset:
                    raise ValueError(
                        "SOUND_CARD_TRIGGER_ID_AFTER_ONSET must be True for"
                        " gapless playback."
                    )
            # make sure the API is one that works with sounddevice
            allowed_apis = ["MME", "Windows WASAPI", "ASIO", None]
            if os.name == "nt" and (params["SOUND_CARD_API"] not in allowed_apis):
                raise ValueError(
                    f"SOUND_CARD_API must be one of {allowed_apis[:-1]} for gapless "
                    f"playback, got {params['SOUND_CARD_API']}."
                )
        elif self.ec._gapless:
            raise RuntimeError(
                'SOUND_CARD_BACKEND must be "sounddevice" for gapless '
                f"playback, got {self.backend_name!r}"
            )
        temp_sound = np.zeros((self._n_channels_tot, 1000))
        temp_sound = self.backend.SoundPlayer(temp_sound, **self._kwargs)
        self.fs = float(temp_sound.fs)
        self._mixer = getattr(temp_sound, "_mixer", None)
        del temp_sound

        if ec._noise_array is None:  # generate AWGN
            # Need to generate at RMS=1 to match TDT circuit, and use a power
            # of 2 length for the RingBuffer (generate >15 seconds)
            n_samples = 2 ** int(np.ceil(np.log2(self.fs * 15)))
            noise = np.random.normal(0, 1.0, (self._n_channels, n_samples))

            # Low-pass if necessary
            if stim_fs < self.fs:
                # note we can use cheap DFT method here b/c
                # circular convolution won't matter for AWGN (yay!)
                freqs = rfftfreq(noise.shape[-1], 1.0 / self.fs)
                noise = rfft(noise, axis=-1)
                noise[:, np.abs(freqs) > stim_fs / 2.0] = 0.0
                noise = irfft(noise, axis=-1)

            # ensure true RMS of 1.0 (DFT method also lowers RMS, compensate here)
            noise /= np.sqrt(np.mean(noise * noise))
            noise = np.concatenate(
                (np.zeros((self._n_channels_stim, noise.shape[1]), noise.dtype), noise)
            )
            self.noise_array = noise
        else:
            self.noise_array = ec._noise_array

            # check data type
            self.noise_array = np.asarray(self.noise_array, dtype=np.float32)

            # check shape and dimensions, make stereo
            self.noise_array = _fix_audio_dims(self.noise_array, self._n_channels)

            # check the length is a power of 2 (required for ringbuffer)
            len_noise_array = self.noise_array.shape[-1]
            if not (
                len_noise_array > 0 and (len_noise_array & (len_noise_array - 1)) == 0
            ):
                raise ValueError(
                    "noise_array must have a length that is a power of 2, "
                    f"got length {len_noise_array}."
                )

            # This limit is currently set by the TDT SerialBuf objects
            # (per channel), it sets the limit on our stimulus durations...
            if np.isclose(ec.stim_fs, 24414, atol=1):
                max_samples = 4000000 - 1
                if self.noise_array.shape[-1] > max_samples:
                    raise RuntimeError(
                        f"Sample too long {self.noise_array.shape[-1]} > {max_samples}"
                    )

        self.noise_level = 0.01
        self.noise = None
        self.audio = None
        self.playing = False
        self._trigger_duration = trigger_duration
        self._trig_scale = trig_scale
        flush_logger()

    def __repr__(self):
        return "<SoundController : %s playback %s trigger ch >" % (
            self._n_channels,
            self._n_channels_stim,
        )

    @property
    def _n_channels_tot(self):
        return self._n_channels_stim + self._n_channels

    def start_noise(self):
        """Start noise."""
        if not self._noise_playing:
            self.noise = self.backend.SoundPlayer(
                self.noise_array * self.noise_level, loop=True, **self._kwargs
            )
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
        if not self.ec._gapless:
            self.stop(wait=False)
            if self.audio is not None:
                self.audio.delete()
                self.audio = None
        if self._n_channels_stim > 0:
            stim = self._make_digital_trigger([1] + self._extra_onset_triggers)
            stim_len = len(stim)
            sample_len = len(samples)
            extra = sample_len - stim_len
            if extra > 0:  # stim shorter than samples (typical)
                stim = np.pad(stim, ((0, extra), (0, 0)), "constant")
            elif extra < 0:  # samples shorter than stim (very brief stim)
                samples = np.pad(samples, ((0, -extra), (0, 0)), "constant")
            # place the drift triggers
            trig2 = self._make_digital_trigger([2])
            trig2_len = trig2.shape[0]
            trig2_starts = []
            for trig2_time in self._drift_trigger_time:
                if trig2_time == "end":
                    stim[-trig2_len:] = np.bitwise_or(stim[-trig2_len:], trig2)
                    trig2_starts += [sample_len - trig2_len]
                else:
                    trig2_start = int(np.round(trig2_time * self.fs))
                    if (trig2_start >= 0 and trig2_start <= stim_len) or (
                        trig2_start < 0 and abs(trig2_start) >= extra
                    ):
                        warnings.warn("Drift triggers overlap with onset triggers.")
                    if (trig2_start > 0 and trig2_start > sample_len - trig2_len) or (
                        trig2_start < 0 and abs(trig2_start) >= sample_len
                    ):
                        warnings.warn(
                            f"Drift trigger at {trig2_time} seconds occurs"
                            " outside stimulus window, "
                            "not stamping "
                            "trigger."
                        )
                        continue
                    stim[trig2_start : trig2_start + trig2_len] = np.bitwise_or(
                        stim[trig2_start : trig2_start + trig2_len], trig2
                    )
                    if trig2_start > 0:
                        trig2_starts += [trig2_start]
                    else:
                        trig2_starts += [sample_len + trig2_start]
            if np.any(np.diff(trig2_starts) < trig2_len):
                warnings.warn(
                    "Some 2-triggers overlap, times should be at "
                    f"least {trig2_len / self.fs} seconds apart."
                )
            self.ec.write_data_line(
                "Drift triggers were stamped at the following times: ",
                str([t2s / self.fs for t2s in trig2_starts]),
            )
            stim = self._scale_digital_trigger(stim)
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
        trigs = np.array(trigs, int)
        assert trigs.ndim == 1
        n_samples = n_each * len(trigs)
        stim = np.zeros((n_samples, self._n_channels_stim), np.int32)
        offset = 0
        for trig in trigs:
            stim[offset : offset + n_on] = trig
            offset += n_each
        return stim

    def _scale_digital_trigger(self, triggers):
        return ((triggers << 8) * self._trig_scale).astype(np.float32)

    def stamp_triggers(
        self, triggers, delay=None, wait_for_last=True, is_trial_id=False
    ):
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
        stim = self._scale_digital_trigger(stim)
        stim = np.pad(stim, ((0, 0), (0, self._n_channels)[self._stim_sl]), "constant")
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
        assert not self.playing or self.ec._gapless
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
        abort_all = getattr(self.backend, "_abort_all_queues", lambda: None)
        abort_all()


def _import_backend(backend):
    # Auto mode is special, will loop through all possible backends
    if backend == "auto":
        backends = list()
        for backend in _BACKENDS:
            try:
                backends.append(_import_backend(backend)[0])
            except Exception:
                pass
        backends = sorted([backend._PRIORITY, backend] for backend in backends)
        if len(backends) == 0:
            raise RuntimeError("Could not load any sound backend: %s" % (_BACKENDS,))
        backend = op.splitext(op.basename(backends[0][1].__file__))[0][1:]
    if backend not in _BACKENDS:
        raise ValueError(
            "Unknown sound card backend %r, must be one of %s"
            % (backend, ("auto",) + _BACKENDS)
        )
    lib = importlib.import_module("._" + backend, package="expyfun._sound_controllers")
    return lib, backend


class SoundPlayer:
    """Play sounds via the sound card."""

    def __new__(self, data, **kwargs):
        """Create a new instance."""
        backend = kwargs.pop("backend", "auto")
        return _import_backend(backend)[0].SoundPlayer(data, **kwargs)
