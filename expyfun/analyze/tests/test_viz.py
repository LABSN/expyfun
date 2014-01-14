import numpy as np
from os import path as op
from nose.tools import assert_raises, assert_equal
import warnings

import expyfun.analyze as ea
from expyfun._utils import _TempDir, requires_pandas

warnings.simplefilter('always')
temp_dir = _TempDir()


@requires_pandas
def test_barplot():
    """Test bar plot function
    """
    tmp = np.arange(12).reshape((3, 4))
    grp1 = np.arange(4).reshape((2, 2))
    grp2 = [[0, 1, 2], [3]]
    assert_raises(TypeError, ea.barplot, tmp, err_bars=True)
    assert_raises(ValueError, ea.barplot, tmp, err_bars='foo')
    ea.barplot(tmp, lines=True, err_bars='sd')
    ea.barplot(tmp, grp1, err_bars='se', group_names=['a', 'b'])
    fname = op.join(temp_dir, 'temp.pdf')
    ea.barplot(tmp, grp2, False, err_bars='ci', group_names=['a', 'b'],
               filename=fname)
    del tmp, grp1, grp2


def test_plot_screen():
    """Test screen plotting function
    """
    tmp = np.ones((10, 20, 2))
    assert_raises(ValueError, ea.plot_screen, tmp)
    tmp = np.ones((10, 20, 3))
    ea.plot_screen(tmp)


def test_format_pval():
    """Test p-value formatting
    """
    foo = ea.format_pval(1e-10, latex=False, scheme='ross')
    assert_equal(foo, 'p < 10^-9')
