# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import os
import platform

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
