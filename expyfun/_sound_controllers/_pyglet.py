"""Pyglet interface for sound output."""

# Authors: Dan McCloy <drmccloy@uw.edu>
#          Eric Larson <larsoner@uw.edu>
#
# License: BSD (3-clause)

import os
import sys
import warnings

import numpy as np

import pyglet

_PRIORITY = 200

_use_silent = (os.getenv('_EXPYFUN_SILENT', '') == 'true')
_opts_dict = dict(linux2=('pulse',),
                  win32=('directsound',),
                  darwin=('openal',))
_opts_dict['linux'] = _opts_dict['linux2']  # new name on Py3k
_driver = _opts_dict[sys.platform] if not _use_silent else ('silent',)

pyglet.options['audio'] = _driver
# We might also want this at some point if we hit OSX problems:
# pyglet.options['shadow_window'] = False

# these must follow the above option setting, so PEP8 complains
try:
    try:
        from pyglet.media.codecs import AudioFormat
    except ImportError:
        from pyglet.media import AudioFormat
    from pyglet.media import Player, SourceGroup  # noqa
    try:
        from pyglet.media.codecs import StaticMemorySource
    except ImportError:
        try:
            from pyglet.media import StaticMemorySource
        except ImportError:
            from pyglet.media.sources.base import StaticMemorySource  # noqa
except Exception as exp:
    warnings.warn('Pyglet could not be imported:\n%s' % exp)
    Player = AudioFormat = SourceGroup = StaticMemorySource = object


def _check_pyglet_audio():
    if pyglet.media.get_audio_driver() is None:
        raise SystemError('pyglet audio ("%s") could not be initialized'
                          % pyglet.options['audio'][0])


class SoundPlayer(Player):
    """SoundPlayer based on Pyglet."""

    def __init__(self, data, fs=None, loop=False, api=None, name=None,
                 fixed_delay=None, api_options=None):
        from .._utils import _new_pyglet
        assert AudioFormat is not None
        if any(x is not None for x in (api, name, fixed_delay, api_options)):
            raise ValueError('The Pyglet backend does not support specifying '
                             'api, name, fixed_delay, or api_options')
        # We could maybe let Pyglet make this decision, but hopefully
        # people won't need to tweak the Pyglet backend anyway
        self.fs = 44100 if fs is None else fs
        super(SoundPlayer, self).__init__()
        _check_pyglet_audio()
        sms = _as_static(data, self.fs)
        if _new_pyglet():
            self.queue(sms)
            self.loop = bool(loop)
        else:
            group = SourceGroup(sms.audio_format, None)
            group.loop = bool(loop)
            group.queue(sms)
            self.queue(group)
        self._ec_duration = sms._duration

    def stop(self, wait=True, extra_delay=0.):
        """Stop."""
        self.pause()
        self.seek(0.)

    @property
    def playing(self):
        # Pyglet has this, but it doesn't notice when it's finished on its own
        return (super(SoundPlayer, self).playing and not
                np.isclose(self.time, self._ec_duration))


def _as_static(data, fs):
    """Get data into the Pyglet audio format."""
    fs = int(fs)
    if data.ndim not in (1, 2):
        raise ValueError('Data must have one or two dimensions')
    n_ch = data.shape[0] if data.ndim == 2 else 1
    audio_format = AudioFormat(channels=n_ch, sample_size=16,
                               sample_rate=fs)
    data = data.T.ravel('C')
    data[data < -1] = -1
    data[data > 1] = 1
    data = (data * (2 ** 15)).astype('int16').tostring()
    return StaticMemorySourceFixed(data, audio_format)


class StaticMemorySourceFixed(StaticMemorySource):
    """Stupid class to fix old Pyglet bug."""

    def __init__(self, data, audio_format):
        self._data = data
        StaticMemorySource.__init__(self, data, audio_format)

    def _get_queue_source(self):
        return self
