from nose.tools import assert_raises, assert_equal
import numpy as np
import warnings

import expyfun.analyze as ea

warnings.simplefilter('always')


def test_dprime():
    assert_raises(TypeError, ea.dprime, 'foo', 0, 0, 0)
    with warnings.catch_warnings(True) as w:
        ea.dprime(1.1, 0, 0, 0)
    assert_equal(len(w), 1)
    assert_equal(0, ea.dprime(1, 0, 1, 0))
    assert_equal(np.inf, ea.dprime(1, 0, 2, 1, False))
    assert_equal(ea.dprime(5, 0, 1, 0), ea.dprime_2afc(5, 1))
