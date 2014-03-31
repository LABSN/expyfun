from nose.tools import assert_true, assert_raises, assert_equal
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
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        # warnings raised only when setting key
        set_config(key, None)
        assert_true(get_config(key) is None)
        assert_raises(KeyError, get_config, key, raise_error=True)
        set_config(key, value)
        assert_equal(get_config(key), value)
        set_config(key, None)
    assert_equal(len(w), 3)
    if old_val is not None:
        os.environ[key] = old_val
    assert_raises(ValueError, get_config, 1)


@deprecated('message')
def deprecated_func():
    pass


@deprecated('message')
class deprecated_class(object):
    def __init__(self):
        pass


def test_deprecated():
    """Test deprecated function
    """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        deprecated_func()
    assert_true(len(w) == 1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        deprecated_class()
    assert_true(len(w) == 1)
