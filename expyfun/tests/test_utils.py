from nose.tools import assert_true, assert_raises
import os
import warnings

from expyfun._utils import get_config, set_config, deprecated

warnings.simplefilter('always')


def test_config():
    """Test expyfun config file support"""
    key = '_EXPYFUN_CONFIG_TESTING'
    value = '123456'
    old_val = os.getenv(key, None)
    os.environ[key] = value
    assert_true(get_config(key) == value)
    del os.environ[key]
    # catch the warning about it being a non-standard config key
    with warnings.catch_warnings(True) as w:
        set_config(key, None)
        assert_true(len(w) == 1)
    assert_true(get_config(key) is None)
    assert_raises(KeyError, get_config, key, raise_error=True)
    set_config(key, value)
    assert_true(get_config(key) == value)
    set_config(key, None)
    if old_val is not None:
        os.environ[key] = old_val


@deprecated('message')
def deprecated_func():
    pass


def test_deprecated():
    """Test deprecated function
    """
    with warnings.catch_warnings(True) as w:
        deprecated_func()
    assert_true(len(w) == 1)
