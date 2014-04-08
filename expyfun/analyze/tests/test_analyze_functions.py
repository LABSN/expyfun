from nose.tools import assert_raises, assert_equal, assert_true
from numpy.testing import assert_allclose
try:
    from scipy.special import logit as splogit
except ImportError:
    splogit = None
import numpy as np
import warnings

import expyfun.analyze as ea

warnings.simplefilter('always')


def test_dprime():
    """Test dprime and dprime_2afc accuracy
    """
    assert_raises(TypeError, ea.dprime, 'foo', 0, 0, 0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        ea.dprime((1.1, 0, 0, 0))
    assert_equal(len(w), 1)
    assert_equal(0, ea.dprime((1, 0, 1, 0)))
    assert_equal(np.inf, ea.dprime((1, 0, 2, 1), False))
    assert_equal(ea.dprime((5, 0, 1, 0)), ea.dprime_2afc((5, 1)))
    assert_raises(ValueError, ea.dprime, np.ones((5, 4, 3)))
    assert_raises(ValueError, ea.dprime, (1, 2, 3))
    assert_raises(ValueError, ea.dprime_2afc, (1, 2, 3))
    assert_equal(np.sum(ea.dprime_2afc([[5, 1], [1, 5]])), 0)


def test_logit():
    """Test logit calculations
    """
    assert_raises(ValueError, ea.logit, 2)
    # On some versions, this throws warnings about divide-by-zero
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        assert_equal(ea.logit(0), -np.inf)
        assert_equal(ea.logit(1), np.inf)
    assert_true(ea.logit(1, max_events=1) < np.inf)
    assert_equal(ea.logit(0.5), 0)
    if splogit is not None:
        # Travis doesn't support scipy.special.logit, but this passes locally:
        foo = np.random.rand(5)
        assert_allclose(ea.logit(foo), splogit(foo))


def test_sigmoid():
    """Test sigmoidal fitting and generation
    """
    n_pts = 1000
    x = np.random.randn(n_pts)
    p0 = (0., 1., 0., 1.)
    y = ea.sigmoid(x, *p0)
    assert_true(np.all(np.logical_and(y <= 1, y >= 0)))
    p = ea.fit_sigmoid(x, y)
    assert_allclose(p, p0, atol=1e-4, rtol=1e-4)

    y += np.random.rand(n_pts) * 0.01
    p = ea.fit_sigmoid(x, y)
    assert_allclose(p, p0, atol=0.1, rtol=0.1)
