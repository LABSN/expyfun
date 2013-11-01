"""Analysis functions (mostly for psychophysics data).
"""

import warnings
import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
from itertools import chain


def dprime(hmfc, zero_correction=True):
    """Estimates d-prime, with optional correction factor to avoid infinites.

    Parameters
    ----------
    hmfc : array-like
        Hits, misses, false-alarms, and correct-rejections, in that order, as a
        four-element list, tuple, or numpy array.  If an Nx4 array is provided,
        it will return an array of dimension (N,).
    zero_correction : bool
        Whether to add a correction factor of 0.5 to each category to prevent
        division-by-zero leading to infinite d-prime values.

    Notes
    -----
    For two-alternative forced-choice tasks, it is recommended to enter correct
    trials as hits and incorrect trials as false alarms, and enter misses and
    correct rejections as 0. An alternative is to use ``dprime_2afc()``, which
    wraps to ``dprime()`` and does this assignment for you.
    """
    vector = False
    hmfc = _check_dprime_inputs(hmfc)
    if len(hmfc.shape) == 1:
        vector = True
        hmfc = np.atleast_2d(hmfc)
    if zero_correction:
        a = 0.5
    else:
        a = 0.0
    dp = ss.norm.ppf((hmfc[:, 0] + a) / (hmfc[:, 0] + hmfc[:, 1] + 2 * a)) \
        - ss.norm.ppf((hmfc[:, 2] + a) / (hmfc[:, 2] + hmfc[:, 3] + 2 * a))
    if vector:
        return dp[0]
    else:
        return dp


def dprime_2afc(hm, zero_correction=True):
    """Estimates d-prime for two-alternative forced-choice paradigms.

    Parameters
    ----------
    hm : array-like
        Correct trials (hits) and incorrect trials (misses), in that order, as
        a two-element list, tuple, or numpy array. If an Nx2 array is provided,
        it will return an array of dimension (N,).
    zero_correction : bool
        Whether to add a correction factor of 0.5 to each category to prevent
        division-by-zero leading to infinite d-prime values.
    """
    hmfc = _check_dprime_inputs(hm, True)
    return dprime(hmfc, zero_correction)


def _check_dprime_inputs(hmfc, tafc=False):
    """Formats input to dprime() and dprime_2afc().

    Parameters
    ----------
    hmfc : array-like
        Hit, miss, false-alarm, correct-rejection; or hit, miss for 2AFC.
    tafc : bool
        Is this a 2AFC design?
    """
    hmfc = np.array(hmfc)
    if len(hmfc.shape) > 2:
        raise ValueError('Argument to dprime() cannot have more than two '
                         'dimensions.')
    elif hmfc.shape[-1] != 2 and tafc:
        raise ValueError('Array dimensions of argument to dprime_2afc() must '
                         'be (2,) or (N, 2).')
    elif hmfc.shape[-1] != 4 and not tafc:
        raise ValueError('Array dimensions of argument to dprime() must be '
                         '(4,) or (N, 4).')

    if len(hmfc.shape) == 1 and tafc:
        hmfc = np.c_[hmfc[0], 0, hmfc[1], 0]
    elif tafc:
        z = np.zeros_like(hmfc[:, 0])
        hmfc = np.c_[hmfc[:, 0], z, hmfc[:, 1], z]
    if hmfc.dtype != np.int64:
        warnings.warn('Argument to dprime() cast to np.int64; floating point '
                      'values will have been truncated.')
        hmfc = hmfc.astype(np.int64)
    return hmfc


def barplot(df, grouping=None, fix_bar_width=True, xlab=None, group_names=None,
            lines=False, err_bars=None, filename=None, bar_kwargs=None,
            err_kwargs=None, line_kwargs=None):
    """Generates optionally grouped barplots with connected line overlays.
    Parameters
    ----------
    df : pandas.DataFrame
        Data to be plotted. If not already a ``DataFrame``, will be coerced to
        one. Passing a ``numpy.ndarray`` as ``df`` should work transparently,
        with sequential integers assigned as column names.
    groups : None | list
        List of lists containing the integers in ``range(len(df.columns))``,
        with sub-lists indicating the desired grouping. For example, if your
        DataFrame has four columns and you want the first bar isolated and the
        remaining three grouped, then specify ``grouping=[[0], [1, 2, 3]]``.
    fix_bar_width : bool
        Should all bars be same width, or all groups be same width?
    xlab : list | None
        Labels for each bar to place along the x-axis. If ``None``, defaults
        to the column names of ``df``.
    group_names : list | None
        Additional labels to go below the individual bar labels on the x-axis.
    lines : bool
        Should lines be plotted over the bars? Values are drawn from the rows
        of ``df``.
    err_bars : str | None
        Type of error bars to be added to the barplot. Possible values are
        ``'sd'`` for sample standard deviation, ``'se'`` for standard error of
        the mean, or ``'ci'`` for 95% confidence interval. If ``None``, no
        error bars will be plotted.
    filename : str
        Full path (absolute or relative) of the output file. At present only
        PDF format implemented.
    bar_kwargs : dict
        arguments passed to ``pyplot.bar()`` (e.g., color, linewidth).
    err_kwargs : dict
        arguments passed to ``pyplot.bar(error_kw)`` (e.g., ecolor, capsize).
    line_kwargs : dict
        arguments passed to ``pyplot.plot()`` (e.g., color, marker, linestyle).

    Returns
    -------
    ax : matplotlib.axes.Axes
        The plot object.
    """
    if err_bars is not None:
        if not isinstance(err_bars, basestring):
            raise TypeError()
        if err_bars not in ['sd', 'se', 'ci']:
            raise ValueError('err_bars must be one of "sd", "se", or "ci".')
    if bar_kwargs is None:
        bar_kwargs = dict()
    if err_kwargs is None:
        err_kwargs = dict()
    if line_kwargs is None:
        line_kwargs = dict()
    if not isinstance(df, pd.core.frame.DataFrame):
        df = pd.DataFrame(df)
    err = None
    wid = 1.0
    gap = 0.2 * wid
    if grouping is None:
        grouping = [range(len(df.columns))]
        fix_bar_width = True
    elif isinstance(grouping, np.ndarray):
        grouping = grouping.tolist()
    gr_flat = list(chain.from_iterable(grouping))
    num_bars = len(gr_flat)
    if xlab is None:
        xlab = np.array(df.columns)[gr_flat].tolist()
    num_groups = len(grouping)
    group_sizes = [len(x) for x in grouping]
    grp_width = [wid * num_bars / float(num_groups) / float(siz)
                 for siz in group_sizes]
    indices = [grouping.index(grp) for grp in grouping
               for bar in xrange(num_bars) if bar in grp]
    if len(set(indices)) == 1:
        offsets = gap * (np.array(indices) + 1)
    else:
        offsets = gap * np.array(indices)
    if fix_bar_width:
        bar_width = [wid for _ in xrange(num_bars)]
        bar_xpos = offsets + np.arange(num_bars)
    else:
        bar_width = [grp_width[x] for x in indices]
        bar_xpos = offsets + np.arange(num_bars) * ([0] + bar_width)[:-1]
    bar_ypos = df.mean()[gr_flat]
    pts_xpos = bar_xpos + 0.5 * np.array(bar_width)
    # basic plot setup
    plt.figure()
    p = plt.subplot(1, 1, 1)
    # error bars
    if err_bars is not None:
        if err_bars == 'sd':  # sample standard deviation
            err = df.std()[gr_flat]
        elif err_bars == 'se':  # standard error of the mean
            err = df.std()[gr_flat] / np.sqrt(len(df.index))
        else:  # 95% confidence interval
            err = 1.96 * df.std()[gr_flat] / np.sqrt(len(df.index))
        bar_kwargs['yerr'] = err
    # barplot & line overlays
    p.bar(bar_xpos, bar_ypos, bar_width, error_kw=err_kwargs, **bar_kwargs)
    if lines:
        for idx in df.index.tolist():
            pts = [df[col][idx]
                   for col in np.array(df.columns)[gr_flat].tolist()]
            p.plot(pts_xpos, pts, **line_kwargs)
    # garnishes
    box_off(p)
    p.tick_params(axis='x', length=0)
    plt.ylabel('d-prime')
    plt.xticks(pts_xpos, xlab)
    p.set_xbound(upper=p.get_xlim()[1] + 0.1)
    # group names
    if group_names is not None:
        gs = np.r_[0, np.cumsum(group_sizes)]
        group_name_pos = [np.mean(pts_xpos[a:b])
                          for a, b in zip(gs[:-1], gs[1:])]
        ypos = p.get_ylim()[0] - 0.1 * np.diff(p.get_ylim())
        for gnp, gn in zip(group_name_pos, group_names):
            p.text(gnp, ypos, gn, ha='center')
    # output file
    if filename is not None:
        plt.savefig(filename, format='pdf', transparent=False)
    plt.draw()
    return plt


def box_off(ax):
    """Remove the top and right edges of a plot frame, and point ticks outward.
    Parameter
    ---------
    ax : matplotlib.axes.Axes
        A matplotlib plot or subplot object.
    """
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis='x', direction='out')
    ax.tick_params(axis='y', direction='out')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
