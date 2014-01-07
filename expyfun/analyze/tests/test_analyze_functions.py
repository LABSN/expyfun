from nose.tools import assert_raises, assert_equal
import numpy as np
import warnings

import expyfun.analyze as ea

warnings.simplefilter('always')


def test_dprime():
    """Test dprime and dprime_2afc accuracy
    """
    assert_raises(TypeError, ea.dprime, 'foo', 0, 0, 0)
    with warnings.catch_warnings(True) as w:
        ea.dprime((1.1, 0, 0, 0))
    assert_equal(len(w), 1)
    assert_equal(0, ea.dprime((1, 0, 1, 0)))
    assert_equal(np.inf, ea.dprime((1, 0, 2, 1), False))
    assert_equal(ea.dprime((5, 0, 1, 0)), ea.dprime_2afc((5, 1)))
    assert_raises(ValueError, ea.dprime, np.ones((5, 4, 3)))
    assert_raises(ValueError, ea.dprime, (1, 2, 3))
    assert_raises(ValueError, ea.dprime_2afc, (1, 2, 3))
    assert_equal(np.sum(ea.dprime_2afc([[5, 1], [1, 5]])), 0)
