import numpy as np
from os import path as op
from nose.tools import assert_raises, assert_equal
import warnings

import expyfun.analyze as ea
from expyfun._utils import _TempDir, requires_pandas, requires_mpl

warnings.simplefilter('always')
temp_dir = _TempDir()


@requires_pandas
def test_barplot_with_pandas():
    import pandas as pd
    tmp = pd.DataFrame(np.arange(20).reshape((4, 5)),
                       columns=['a', 'b', 'c', 'd', 'e'],
                       index=['one', 'two', 'three', 'four'])
    ea.barplot(tmp)
    ea.barplot(tmp, axis=0, lines=True)


@requires_mpl
def test_barplot():
    """Test bar plot function
    """
    import matplotlib.pyplot as plt
    ax = plt.subplot(1, 1, 1)
    tmp1 = np.arange(4)  # 1-Dim
    tmp2 = np.arange(20).reshape((4, 5))  # 2-Dim
    ea.barplot(tmp1, err_bars=tmp1, brackets=[(0, 1), (2, 3)],
               bracket_text=['foo', 'bar'], ax=ax)
    ea.barplot(tmp1, groups=[[0, 1, 2], [3]], eq_group_widths=True,
               brackets=[([0], 3)], bracket_text=['foo'])
    ea.barplot(tmp2, lines=True, ylim=(0, 2), err_bars='se')
    ea.barplot(tmp2, groups=[[0, 1], [2, 3]], err_bars='ci',
               group_names=['foo', 'bar'])
    extns = ['eps', 'jpg', 'pdf', 'png', 'raw', 'svg', 'tif']
    for ext in extns:
        fname = op.join(temp_dir, 'temp.' + ext)
        ea.barplot(tmp2, groups=[[0, 1, 2], [3]], err_bars='sd', fname=fname)
    assert_raises(ValueError, ea.barplot, np.arange(8).reshape((2, 2, 2)))
    assert_raises(ValueError, ea.barplot, tmp2, err_bars='foo')
    assert_raises(ValueError, ea.barplot, tmp2, gap_size=1.1)
    assert_raises(ValueError, ea.barplot, tmp1, err_bars=np.arange(3))
    assert_raises(ValueError, ea.barplot, tmp1, err_bars='sd')
    assert_raises(ValueError, ea.barplot, tmp1, brackets=[(0, 1)],
                  bracket_text=['foo', 'bar'])
    assert_raises(ValueError, ea.barplot, tmp1, brackets=[(1,)],
                  bracket_text=['foo'])


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
