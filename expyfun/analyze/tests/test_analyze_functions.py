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


def test_plotting():
    """
    """
    tmp = np.arange(12).reshape((3, 4))
    grp1 = np.arange(4).reshape((2, 2))
    grp2 = [[0, 1, 2], [3]]
    assert_raises(TypeError, ea.barplot, tmp, err_bars=True)
    assert_raises(ValueError, ea.barplot, tmp, err_bars='foo')
    ea.barplot(tmp, lines=True, err_bars='sd')
    ea.barplot(tmp, grp1, err_bars='se', group_names=['a', 'b'])
    ea.barplot(tmp, grp2, False, err_bars='ci', group_names=['a', 'b'])
    del tmp, grp1, grp2
