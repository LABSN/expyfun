import os
import warnings

import numpy as np
import pytest

from expyfun._utils import _fix_audio_dims, deprecated, get_config, set_config

warnings.simplefilter("always")


def test_config():
    """Test expyfun config file support."""
    key = "_EXPYFUN_CONFIG_TESTING"
    value = "123456"
    old_val = os.getenv(key, None)
    os.environ[key] = value
    assert get_config(key) == value
    del os.environ[key]
    # catch the warning about it being a non-standard config key
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # warnings raised only when setting key
        set_config(key, None)
        assert get_config(key) is None
        pytest.raises(KeyError, get_config, key, raise_error=True)
        set_config(key, value)
        assert get_config(key) == value
        set_config(key, None)
    assert len(w) == 3
    if old_val is not None:
        os.environ[key] = old_val
    pytest.raises(ValueError, get_config, 1)
    get_config(None)
    set_config(None, "0")


@deprecated("message")
def deprecated_func():
    """Deprecated function."""
    pass


@deprecated("message")
class deprecated_class:
    """Deprecated class."""

    def __init__(self):
        pass


def test_deprecated():
    """Test deprecated function."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        deprecated_func()
    assert len(w) == 1
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        deprecated_class()
    assert len(w) == 1


def test_audio_dims():
    """Test audio dimension fixing."""
    n_samples = 10
    x = range(n_samples)
    y = _fix_audio_dims(x, 1)
    assert y.shape == (1, n_samples)
    # tiling for stereo
    y = _fix_audio_dims(x, 2)
    assert y.shape == (2, n_samples)
    y = _fix_audio_dims(y, 2)
    assert y.shape == (2, n_samples)
    # no tiling for >2 channel output
    with pytest.raises(ValueError, match="channel count 1 did not .* 3"):
        _fix_audio_dims(x, 3)
    for dim in (1, 3):
        want = "signal channel count 2 did not match required channel count %s" % dim
        with pytest.raises(ValueError, match=want):
            _fix_audio_dims(y, dim)
    for n_channels in (1, 2, 3):
        with pytest.raises(ValueError, match="must have one or two dimension"):
            _fix_audio_dims(np.zeros((2, 2, 2)), n_channels)
