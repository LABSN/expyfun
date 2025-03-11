import warnings
from os import path as op

import numpy as np
import pytest
from numpy.testing import assert_equal

import expyfun.analyze as ea
from expyfun._utils import _TempDir, requires_lib

temp_dir = _TempDir()


def _check_warnings(w):
    """Silly helper to deal with MPL deprecation warnings."""
    assert all(["expyfun" not in ww.filename for ww in w])


@requires_lib("pandas")
def test_barplot_with_pandas():
    """Test bar plot function pandas support."""
    import pandas as pd

    tmp = pd.DataFrame(
        np.arange(20).reshape((4, 5)),
        columns=["a", "b", "c", "d", "e"],
        index=["one", "two", "three", "four"],
    )
    ea.barplot(tmp)
    ea.barplot(tmp, axis=0, lines=True)


@pytest.fixture
def tmp_err():  # noqa
    rng = np.random.RandomState(0)
    tmp = np.ones(4) + rng.rand(4)
    err = 0.1 + tmp / 5.0
    return tmp, err


def test_barplot_degenerate(tmp_err):
    """Test bar plot degenerate cases."""
    import matplotlib.pyplot as plt

    tmp, err = tmp_err
    # too many data dimensions:
    pytest.raises(ValueError, ea.barplot, np.arange(8).reshape((2, 2, 2)))
    # gap_size > 1:
    pytest.raises(ValueError, ea.barplot, tmp, gap_size=1.1)
    # shape mismatch between data & error bars:
    pytest.raises(ValueError, ea.barplot, tmp, err_bars=np.arange(3))
    # bad err_bar string:
    pytest.raises(ValueError, ea.barplot, tmp, err_bars="foo")
    # cannot calculate 'sd' error bars with only 1 value per bar:
    pytest.raises(ValueError, ea.barplot, tmp, err_bars="sd")
    # mismatched lengths of brackets & bracket_text:
    pytest.raises(
        ValueError, ea.barplot, tmp, brackets=[(0, 1)], bracket_text=["foo", "bar"]
    )
    # bad bracket spec:
    pytest.raises(ValueError, ea.barplot, tmp, brackets=[(1,)], bracket_text=["foo"])
    plt.close("all")


def test_barplot_single(tmp_err):
    """Test with single data point and single error bar spec."""
    import matplotlib.pyplot as plt

    tmp, err = tmp_err
    ea.barplot(2, err_bars=0.2)
    plt.close("all")


@pytest.mark.timeout(15)
def test_barplot_single_spec(tmp_err):
    """Test with one data point per bar and user-specified err ranges."""
    import matplotlib.pyplot as plt

    tmp, err = tmp_err
    _, axs = plt.subplots(1, 5, sharey=False)
    ea.barplot(
        tmp,
        err_bars=err,
        brackets=([2, 3], [0, 1]),
        ax=axs[0],
        bracket_text=["foo", "bar"],
        bracket_inline=True,
    )
    ea.barplot(
        tmp,
        err_bars=err,
        brackets=((0, 2), (1, 3)),
        ax=axs[1],
        bracket_text=["foo", "bar"],
    )
    ea.barplot(
        tmp,
        err_bars=err,
        brackets=[[2, 1], [0, 3]],
        ax=axs[2],
        bracket_text=["foo", "bar"],
    )
    ea.barplot(
        tmp,
        err_bars=err,
        brackets=[(0, 1), (0, 2), (0, 3)],
        bracket_text=["foo", "bar", "baz"],
        ax=axs[3],
    )
    ea.barplot(
        tmp,
        err_bars=err,
        brackets=[(0, 1), (2, 3), (0, 2), (1, 3)],
        bracket_text=["foo", "bar", "baz", "snafu"],
        ax=axs[4],
    )
    ea.barplot(
        tmp,
        groups=[[0, 1, 2], [3]],
        eq_group_widths=True,
        brackets=[(0, 1), (1, 2), ([0, 1, 2], 3)],
        bracket_text=["foo", "bar", "baz"],
        bracket_group_lines=True,
    )
    plt.close("all")


@pytest.mark.timeout(10)
def test_barplot_multiple():
    """Test with multiple data points per bar and calculated ranges."""
    import matplotlib.pyplot as plt

    rng = np.random.RandomState(0)
    tmp = (rng.randn(20) + np.arange(20)).reshape((5, 4))  # 2-dim
    _, axs = plt.subplots(1, 4, sharey=False)
    ea.barplot(tmp, lines=True, err_bars="sd", ax=axs[0], smart_defaults=False)
    ea.barplot(tmp, lines=True, err_bars="ci", ax=axs[1], axis=0)
    ea.barplot(tmp, lines=True, err_bars="se", ax=axs[2], ylim=(0, 30))
    ea.barplot(
        tmp,
        lines=True,
        err_bars="se",
        ax=axs[3],
        groups=[[0, 1, 2], [3, 4]],
        bracket_group_lines=True,
        brackets=[(0, 1), (1, 2), (3, 4), ([0, 1, 2], [3, 4])],
        bracket_text=["foo", "bar", "baz", "snafu"],
    )
    extns = ["pdf"]  # jpg, tif not supported; 'png', 'raw', 'svg' not tested
    for ext in extns:
        fname = op.join(temp_dir, "temp." + ext)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ea.barplot(tmp, groups=[[0, 1, 2], [3]], err_bars="sd", axis=0, fname=fname)
            plt.close()
        _check_warnings(w)
    plt.close("all")


def test_plot_screen():
    """Test screen plotting function."""
    tmp = np.ones((10, 20, 2))
    pytest.raises(ValueError, ea.plot_screen, tmp)
    tmp = np.ones((10, 20, 3))
    ea.plot_screen(tmp)


def test_format_pval():
    """Test p-value formatting."""
    foo = ea.format_pval(1e-10, latex=False)
    bar = ea.format_pval(1e-10, scheme="ross")
    baz = ea.format_pval([0.2, 0.02])
    qux = ea.format_pval(0.002, scheme="stars")
    assert_equal(foo, "p < 10^-9")
    assert_equal(bar, "$p < 10^{{-9}}$")
    assert_equal(baz[0], "$n.s.$")
    assert_equal(qux, "${*}{*}$")
