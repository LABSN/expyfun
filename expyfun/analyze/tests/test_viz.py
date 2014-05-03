import numpy as np
from os import path as op
from nose.tools import assert_raises, assert_equal
import warnings

import expyfun.analyze as ea
from expyfun._utils import _TempDir, requires_pandas

warnings.simplefilter('always')
temp_dir = _TempDir()


@requires_pandas
def test_barplot_with_pandas():
    """Test bar plot function pandas support"""
    import pandas as pd
    tmp = pd.DataFrame(np.arange(20).reshape((4, 5)),
                       columns=['a', 'b', 'c', 'd', 'e'],
                       index=['one', 'two', 'three', 'four'])
    ea.barplot(tmp)
    ea.barplot(tmp, axis=0, lines=True)


def test_barplot():
    """Test bar plot function
    """
    import matplotlib.pyplot as plt
    ea.barplot(2, err_bars=0.2)  # 0-dim

    tmp = np.ones(4) + np.random.rand(4)  # 1-dim
    err = 0.1 + tmp / 5.
    _, axs = plt.subplots(1, 5, sharey=True)
    ea.barplot(tmp, err_bars=err, brackets=[(2, 3), (0, 1)], ax=axs[0],
               bracket_text=['foo', 'bar'])
    ea.barplot(tmp, err_bars=err, brackets=[(0, 2), (1, 3)], ax=axs[1],
               bracket_text=['foo', 'bar'])
    ea.barplot(tmp, err_bars=err, brackets=[(2, 1), (0, 3)], ax=axs[2],
               bracket_text=['foo', 'bar'])
    ea.barplot(tmp, err_bars=err, brackets=[(0, 1), (0, 2), (0, 3)],
               bracket_text=['foo', 'bar', 'baz'], ax=axs[3])
    ea.barplot(tmp, err_bars=err, brackets=[(0, 1), (2, 3), (0, 2), (1, 3)],
               bracket_text=['foo', 'bar', 'baz', 'snafu'], ax=axs[4])
    ea.barplot(tmp, groups=[[0, 1, 2], [3]], eq_group_widths=True,
               brackets=[(0, 1), (1, 2), ([0, 1, 2], 3)],
               bracket_text=['foo', 'bar', 'baz'],
               bracket_group_lines=True)
    assert_raises(ValueError, ea.barplot, tmp, err_bars=np.arange(3))
    assert_raises(ValueError, ea.barplot, tmp, err_bars='sd')
    assert_raises(ValueError, ea.barplot, tmp, brackets=[(0, 1)],
                  bracket_text=['foo', 'bar'])
    assert_raises(ValueError, ea.barplot, tmp, brackets=[(1,)],
                  bracket_text=['foo'])
    tmp = (np.random.randn(20) + np.arange(20)).reshape((5, 4))  # 2-dim
    _, axs = plt.subplots(1, 4, sharey=False)
    ea.barplot(tmp, lines=True, err_bars='sd', ax=axs[0], smart_defaults=False)
    ea.barplot(tmp, lines=True, err_bars='ci', ax=axs[1], axis=0)
    ea.barplot(tmp, lines=True, err_bars='se', ax=axs[2], ylim=(0, 30))
    ea.barplot(tmp, lines=True, err_bars='se', ax=axs[3],
               groups=[[0, 1, 2], [3, 4]], bracket_group_lines=True,
               brackets=[(0, 1), (1, 2), (3, 4), ([0, 1, 2], [3, 4])],
               bracket_text=['foo', 'bar', 'baz', 'snafu'])
    extns = ['eps', 'pdf', 'png', 'raw', 'svg']  # jpg, tif not supported
    for ext in extns:
        fname = op.join(temp_dir, 'temp.' + ext)
        ea.barplot(tmp, groups=[[0, 1, 2], [3]], err_bars='sd', axis=0,
                   fname=fname)
    assert_raises(ValueError, ea.barplot, np.arange(8).reshape((2, 2, 2)))
    assert_raises(ValueError, ea.barplot, tmp, err_bars='foo')
    assert_raises(ValueError, ea.barplot, tmp, gap_size=1.1)


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
    bar = ea.format_pval(1e-10, scheme='ross')
    baz = ea.format_pval([0.2, 0.02])
    assert_equal(foo, 'p < 10^-9')
    assert_equal(bar, '$p < 10^{{-9}}$')
    assert_equal(baz[0], '$n.s.$')
