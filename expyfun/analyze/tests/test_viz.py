import numpy as np
import pandas as pd
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
    grp1 = np.arange(4).reshape((2, 2))
    grp2 = [[0, 1, 2], [3]]
    tmp1 = np.arange(20).reshape((4, 5))
    tmp2 = pd.DataFrame(tmp1, columns=['a', 'b', 'c', 'd', 'e'],
                        index=['one', 'two', 'three', 'four'])
    tmp3 = np.arange(4)
    ea.barplot(tmp2, axis=0, lines=True, err_bars='sd',
               brackets=[(0, 1), (2, 3)], bracket_text=['foo', 'bar'])
    ea.barplot(tmp2, err_bars='se', groups=grp1,
               brackets=[([0], 2)], bracket_text=['foo'])
    ea.barplot(tmp1, groups=grp1, err_bars='ci', group_names=['a', 'b'])
    ea.barplot(tmp3, groups=grp2, eq_group_widths=True, err_bars=tmp3)
    extns = ['eps', 'jpg', 'pdf', 'png', 'raw', 'svg', 'tif']
    for ext in extns:
        fname = op.join(temp_dir, 'temp.' + ext)
        ea.barplot(tmp2, groups=grp2, err_bars='ci', filename=fname)
    assert_raises(ValueError, ea.barplot, tmp1, gap_size=1.1)
    assert_raises(ValueError, ea.barplot, tmp1, err_bars='foo')
    assert_raises(ValueError, ea.barplot, np.arange(8).reshape((2, 2, 2)))
    assert_raises(ValueError, ea.barplot, np.arange(4), err_bars=np.arange(3))


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
    foo = ea.format_pval(1e-10, latex=False)
    bar = ea.format_pval(1e-10, latex=True, scheme='ross')
    assert_equal(foo, 'p < 10^-9')
    assert_equal(bar, '$p < 10^{{-9}}$')
