# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import os
import platform
import time

import pytest

from expyfun._sound_controllers import _AUTO_BACKENDS
from expyfun._utils import _get_display

# Unknown pytest problem with readline<->deprecated decorator
try:
    import readline  # noqa
except Exception:
    pass


@pytest.fixture(scope="session")
def matplotlib_config():
    """Configure matplotlib for viz tests."""
    import matplotlib

    matplotlib.use("agg")  # don't pop up windows
    import matplotlib.pyplot as plt

    assert plt.get_backend() == "agg"
    # overwrite some params that can horribly slow down tests that
    # users might have changed locally (but should not otherwise affect
    # functionality)
    plt.ioff()
    plt.rcParams["figure.dpi"] = 100
    os.environ["_EXPYFUN_WIN_INVISIBLE"] = "true"


@pytest.fixture(scope="function")
def hide_window():
    """Hide the expyfun window."""
    try:
        _get_display()
    except Exception as exp:
        pytest.skip("Windowing unavailable (%s)" % exp)


@pytest.fixture(autouse=True)
def _reset_macos_audio_buffer(monkeypatch):
    """Undo our 200 ms sounddevice blocks on macOS after a test uses them.

    The device I/O buffer size is per process on macOS, and while it is large,
    Apple's OpenAL (pyglet backend) can't reuse buffers after a stop/seek
    ("Failed to add data to buffer"). This only matters when both backends are
    used in one process, so briefly run a default-size stream to reset it.
    """
    if platform.system() != "Darwin":
        yield
        return
    try:
        from expyfun._sound_controllers import _sounddevice
    except Exception:  # sounddevice not installed
        yield
        return
    used = list()
    orig = _sounddevice._init_stream
    monkeypatch.setattr(
        _sounddevice, "_init_stream", lambda *a, **k: used.append(1) or orig(*a, **k)
    )
    yield
    if used:
        _sounddevice._abort_all_queues()
        stream = _sounddevice.sd.RawOutputStream(
            channels=1, dtype="float32", latency="low", callback=_silence
        )
        with stream:
            time.sleep(0.1)


def _silence(outdata, frames, t, status):
    outdata[:] = bytes(len(outdata))


_SOUND_CARD_ACS = tuple(
    {"TYPE": "sound_card", "SOUND_CARD_BACKEND": backend} for backend in _AUTO_BACKENDS
)
_SOUND_CARD_PARAMS = [pytest.param("tdt", id="tdt")]
for val in _SOUND_CARD_ACS:
    if val["SOUND_CARD_BACKEND"] == "pyglet":
        val.update(
            SOUND_CARD_API=None, SOUND_CARD_NAME=None, SOUND_CARD_FIXED_DELAY=None
        )
    elif val["SOUND_CARD_BACKEND"] == "sounddevice":
        val.update(SOUND_CARD_TRIGGER_ID_AFTER_ONSET=True)
        if platform.system() == "Windows":
            val.update(SOUND_CARD_API="MME", SOUND_CARD_NAME="Sound Mapper")
    _SOUND_CARD_PARAMS.append(pytest.param(val, id=f"{val['SOUND_CARD_BACKEND']}"))


@pytest.fixture(scope="module", params=tuple(_SOUND_CARD_PARAMS))
def ac(request):
    """Get the backend name."""
    param = request.param
    if isinstance(param, dict):
        param = param.copy()  # make sure we don't errantly modify inplace
    yield param
