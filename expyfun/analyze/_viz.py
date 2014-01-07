"""Analysis visualization functions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain


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


def plot_screen(screen, ax=None):
    """Plot a captured screenshot

    Parameters
    ----------
    screen : array
        The N x M x 3 (or 4) array of screen pixel values.
    ax : matplotlib Axes | None
        If provided, the axes will be plotted to and cleared of ticks.
        If None, a figure will be created.

    Retruns
    -------
    ax : matplotlib Axes
        The axes used to plot the image.
    """
    screen = np.array(screen)
    if screen.ndim != 3 or screen.shape[2] not in [3, 4]:
        raise ValueError('screen must be a 3D array with 3 or 4 channels')
    if ax is None:
        plt.figure()
        ax = plt.axes([0, 0, 1, 1])
    ax.imshow(screen)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.box('off')
    return ax
