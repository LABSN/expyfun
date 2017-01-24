from nose.tools import assert_raises, assert_equal, assert_true
from numpy.testing import assert_allclose, assert_array_equal
try:
    from scipy.special import logit as splogit
except ImportError:
    splogit = None
import numpy as np
import warnings

import expyfun.analyze as ea

warnings.simplefilter('always')


def assert_rts_equal(actual, desired):
    """Helper to assert RTs are equal."""
    assert_true(isinstance(actual, tuple))
    assert_true(isinstance(desired, (list, tuple)))
    assert_equal(len(actual), 2)
    assert_equal(len(desired), 2)
    kinds = ['hits', 'fas']
    for act, des, kind in zip(actual, desired, kinds):
        assert_allclose(act, des, atol=1e-7,
                        err_msg='{0} mismatch'.format(kind))


def assert_hmfc(presses, targets, foils, hmfco, rts, tmin=0.1, tmax=0.6):
    """Assert HMFC is correct."""
    out = ea.press_times_to_hmfc(presses, targets, foils, tmin, tmax)
    assert_array_equal(out, hmfco)
    out = ea.press_times_to_hmfc(presses, targets, foils, tmin, tmax)
    assert_array_equal(out, hmfco)
    out = ea.press_times_to_hmfc(presses, targets, foils, tmin, tmax,
                                 return_type=['counts', 'rts'])
    assert_array_equal(out[0][:4:2], list(map(len, out[1])))
    assert_array_equal(out[0], hmfco)
    assert_rts_equal(out[1], rts)
    # reversing targets and foils
    out = ea.press_times_to_hmfc(presses, foils, targets, tmin, tmax,
                                 return_type=['counts', 'rts'])
    assert_array_equal(out[0], np.array(hmfco)[[2, 3, 0, 1, 4]])
    assert_rts_equal(out[1], rts[::-1])


def test_presses_to_hmfc():
    """Test converting press times to HMFCO and RTs."""
    # Simple example
    targets = [0., 1.]
    foils = [0.5, 1.5]

    presses = [0.1, 1.6]  # presses right at tmin/tmax
    hmfco = [2, 0, 0, 2, 0]
    rts = [[0.1, 0.6], []]
    assert_hmfc(presses, targets, foils, hmfco, rts)

    presses = [0.65, 1.601]  # just past the boundary
    hmfco = [0, 2, 2, 0, 0]
    rts = [[], [0.15, 0.101]]
    assert_hmfc(presses, targets, foils, hmfco, rts)

    presses = [0.75, 1.55]  # smaller than tmin
    hmfco = [1, 1, 1, 1, 0]
    rts = [[0.55], [0.25]]
    assert_hmfc(presses, targets, foils, hmfco, rts)

    presses = [0.76, 2.11]  # greater than tmax
    hmfco = [0, 2, 1, 1, 1]
    rts = [[], [0.26]]
    assert_hmfc(presses, targets, foils, hmfco, rts)

    # A complicated example: multiple preses to targ
    targets = [0, 2, 3]
    foils = [1, 4]
    tmin, tmax = 0., 0.5
    presses = [0.111, 0.2, 1.101, 1.3, 2.222, 2.333, 2.7, 5.]
    hmfco = [2, 1, 1, 1, 2]
    rts = [[0.111, 0.222], [0.101]]
    assert_hmfc(presses, targets, foils, hmfco, rts)

    presses = []  # no presses
    hmfco = [0, 3, 0, 2, 0]
    rts = [[], []]
    assert_hmfc(presses, targets, foils, hmfco, rts)

    presses = [-1, 7, 8]  # all errant presses
    hmfco = [0, 3, 0, 2, 3]
    rts = [[], []]
    assert_hmfc(presses, targets, foils, hmfco, rts)

    # lots of presses
    targets = [1, 2, 5, 6, 7]
    foils = [0, 3, 4, 8]
    presses = [0.201, 2.101, 4.202, 5.102, 6.103, 10.]
    hmfco = [3, 2, 2, 2, 1]
    rts = [[0.101, 0.102, 0.103], [0.201, 0.202]]
    assert_hmfc(presses, targets, foils, hmfco, rts)

    # Bad inputs
    assert_raises(ValueError, ea.press_times_to_hmfc,
                  presses, targets, foils, tmin, 1.1)
    assert_raises(ValueError, ea.press_times_to_hmfc,
                  presses, targets, foils, tmin, tmax, 'foo')


def test_dprime():
    """Test dprime and dprime_2afc accuracy."""
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
    # test simple larger dimensionality support
    assert_equal(ea.dprime((5, 0, 1, 0)), ea.dprime([[[5, 0, 1, 0]]])[0][0])


def test_logit():
    """Test logit calculations."""
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
    foo = np.array([[0, 0.5, 1], [1, 0.5, 0]])
    bar = np.ones_like(foo).astype(int)
    assert_true(np.all(np.equal(ea.logit(foo, 1), np.zeros_like(foo))))
    assert_true(np.all(np.equal(ea.logit(foo, [1, 1, 1]), np.zeros_like(foo))))
    assert_true(np.all(np.equal(ea.logit(foo, bar), np.zeros_like(foo))))
    assert_raises(ValueError, ea.logit, foo, [1, 1])  # can't broadcast


def test_sigmoid():
    """Test sigmoidal fitting and generation."""
    n_pts = 1000
    x = np.random.randn(n_pts)
    p0 = (0., 1., 0., 1.)
    y = ea.sigmoid(x, *p0)
    assert_true(np.all(np.logical_and(y <= 1, y >= 0)))
    p = ea.fit_sigmoid(x, y)
    assert_allclose(p, p0, atol=1e-4, rtol=1e-4)
    p = ea.fit_sigmoid(x, y, (0, 1, None, None), ('upper', 'lower'))
    assert_allclose(p, p0, atol=1e-4, rtol=1e-4)

    y += np.random.rand(n_pts) * 0.01
    p = ea.fit_sigmoid(x, y)
    assert_allclose(p, p0, atol=0.1, rtol=0.1)


def test_rt_chisq():
    """Test reaction time chi-square fitting."""
    # 1D should return single float
    foo = np.random.rand(30)
    assert_raises(ValueError, ea.rt_chisq, foo - 1.)
    assert_equal(np.array(ea.rt_chisq(foo)).shape, ())
    # 2D should return array with shape of input but without ``axis`` dimension
    foo = np.random.rand(30).reshape((2, 3, 5))
    for axis in range(-1, foo.ndim):
        bar = ea.rt_chisq(foo, axis=axis)
        assert_true(np.all(np.equal(np.delete(foo.shape, axis),
                                    np.array(bar.shape))))
    foo_bad = np.concatenate((np.random.rand(30), [100.]))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        bar = ea.rt_chisq(foo_bad)
    assert_equal(len(w), 1)  # warn that there was a likely bad value
