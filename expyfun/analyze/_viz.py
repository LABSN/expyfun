"""Analysis visualization functions
"""

import numpy as np
from itertools import chain
try:
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
except ImportError:
    plt = None
try:
    from pandas.core.frame import DataFrame
except ImportError:
    DataFrame = None

from .._utils import string_types


def format_pval(pval, latex=True, scheme='default'):
    """Format a p-value using one of several schemes.

    Parameters
    ----------
    pval : float | array-like
        The raw p-value(s).
    latex : bool
        Whether to use LaTeX wrappers suitable for use with matplotlib.
    scheme : str
        A keyword indicating the formatting scheme. Currently supports "stars",
        "ross", and "default"; any other string will yield the same as
        "default".

    Returns
    -------
    pv : str | np.objectarray
        A string or array of strings of formatted p-values. If a list output is
        preferred, users may call ``.tolist()`` on the output of the function.
    """
    single_value = False
    if np.array(pval).shape == ():
        single_value = True
    pval = np.atleast_1d(np.asanyarray(pval))
    # add a tiny amount to handle cases where p is exactly a power of ten
    pval = pval + np.finfo(pval.dtype).eps
    expon = np.trunc(np.log10(pval)).astype(int)  # exponents
    pv = np.zeros_like(pval, dtype=object)
    if latex:
        wrap = '$'
        brac = '{{'
        brak = '}}'
    else:
        wrap = ''
        brac = ''
        brak = ''
    if scheme == 'ross':  # (exact value up to 4 decimal places)
        pv[pval >= 0.0001] = [wrap + 'p = {:.4f}'.format(x) + wrap
                              for x in pval[pval > 0.0001]]
        pv[pval < 0.0001] = [wrap + 'p < 10^' + brac + '{}'.format(x) + brak +
                             wrap for x in expon[pval < 0.0001]]
    elif scheme == 'stars':
        star = '{*}' if latex else '*'
        pv[pval >= 0.05] = wrap + '' + wrap
        pv[pval < 0.05] = wrap + star + wrap
        pv[pval < 0.01] = wrap + star * 2 + wrap
        pv[pval < 0.001] = wrap + star * 3 + wrap
    else:  # scheme == 'default'
        pv[pval >= 0.05] = wrap + 'n.s.' + wrap
        pv[pval < 0.05] = wrap + 'p < 0.05' + wrap
        pv[pval < 0.01] = wrap + 'p < 0.01' + wrap
        pv[pval < 0.001] = wrap + 'p < 0.001' + wrap
        pv[pval < 0.0001] = [wrap + 'p < 10^' + brac + '{}'.format(x) + brak +
                             wrap for x in expon[pval < 0.0001]]
    if single_value:
        pv = pv[0]
    return(pv)


def barplot(h, axis=-1, ylim=None, err_bars=None, lines=False,
            groups=None, eq_group_widths=False, gap_size=0.2,
            brackets=None, bracket_text=None, bracket_group_lines=False,
            bar_names=None, group_names=None, bar_kwargs=None,
            err_kwargs=None, line_kwargs=None, bracket_kwargs=None,
            figure_kwargs=None, smart_defaults=True, fname=None, ax=None):
    """Makes barplots w/ optional line overlays, grouping, & signif. brackets.

    Parameters
    ----------
    h : array-like
        If ``h`` is 2-dimensional, heights will be calculated as means along
        the axis given by ``axis``. If ``h`` is of lower dimension, it is
        treated as raw height values. If ``h`` is a pandas ``DataFrame`` and
        ``bar_names`` is None, ``bar_names`` will be inferred from the
        ``DataFrame``'s ``column`` labels (if ``axis=0``) or ``index`` labels.
    axis : int
        The axis along which to calculate mean values to determine bar heights.
        Ignored if ``h`` is 0- or 1-dimensional.
    ylim : tuple | None
        y-axis limits passed to ``matplotlib.pyplot.subplot.set_ylim()``.
    err_bars : str | array-like | None
        Type of error bars to be added to the barplot. Possible values are
        ``'sd'`` for sample standard deviation, ``'se'`` for standard error of
        the mean, or ``'ci'`` for 95% confidence interval. If ``None``, no
        error bars will be plotted. Custom error bar heights are possible by
        passing an array-like object; in such cases ``err_bars`` must have the
        same dimensionality and shape as ``h``.
    lines : bool
        Whether to plot within-subject data as lines overlaid on the barplot.
    groups : list | None
        List of lists containing the integers in ``range(num_bars)``, with
        sub-lists indicating the desired grouping. For example, if ``h`` has
        has shape (10, 4) and ``axis = -1`` then "num_bars" is 4; if you want
        the first bar isolated and the remaining three grouped, then specify
        ``groups=[[0], [1, 2, 3]]``.
    eq_group_widths : bool
        Should all groups have the same width? If ``False``, all bars will have
        the same width. Ignored if ``groups=None``, since the bar/group
        distinction is meaningless in that case.
    gap_size : float
        Width of the gap between groups (if ``eq_group_width = True``) or
        between bars, expressed as a proportion [0,1) of group or bar width.
    brackets : list of tuples | None
        Location of significance brackets. Scheme is similar to ``grouping``;
        if you want a bracket between the first and second bar and another
        between the third and fourth bars, specify as [(0, 1), (2, 3)]. If you
        want brackets between groups of bars instead of between bars, indicate
        the groups as lists within the tuple: [([0, 1], [2, 3])].
        For best results, pairs of adjacent bars should come earlier in the
        list than non-adjacent pairs.
    bracket_text : str | list | None
        Text to display above brackets.
    bracket_group_lines : bool
        When drawing brackets between groups rather than single bars, should a
        horizontal line be drawn at each foot of the bracket to indicate this?
    bar_names : array-like | None
        Optional axis labels for each bar.
    group_names : array-like | None
        Optional axis labels for each group.
    bar_kwargs : dict
        Arguments passed to ``matplotlib.pyplot.bar()`` (ex: color, linewidth).
    err_kwargs : dict
        Arguments passed to ``matplotlib.pyplot.bar(error_kw)`` (ex: ecolor,
        capsize).
    line_kwargs : dict
        Arguments passed to ``matplotlib.pyplot.plot()`` (e.g., color, marker,
        linestyle).
    bracket_kwargs : dict
        arguments passed to ``matplotlib.pyplot.plot()`` (e.g., color, marker,
        linestyle).
    figure_kwargs : dict
        arguments passed to ``matplotlib.pyplot.figure()`` (e.g., figsize, dpi,
        frameon).
    smart_defaults : bool
        Whether to use pyplot default colors (``False``), or something more
        pleasing to the eye (``True``).
    fname : str | None
        Path and name of output file. Type is inferred from ``fname`` and
        should work for any of the types supported by pyplot (pdf, eps,
        svg, png, raw).
    ax : matplotlib.pyplot.axes | None
        A ``matplotlib.pyplot.axes`` instance.  If none, a new figure with a
        single subplot will be created.

    Returns
    -------
    p : handle for the ``matplotlib.pyplot.subplot`` instance.
    b : handle for the ``matplotlib.pyplot.bar`` instance.

    Notes
    -----
    Known limitations:
      1 Bracket heights don't get properly set when generating multiple
        subplots with ``sharey=True`` (matplotlib seems to temporarily force
        the ``ylim`` to +/- 0.6 in this case). Work around is to use
        ``sharey=False`` and manually set ``ylim`` for each subplot.
      2 Brackets that span groups cannot span partial groups. For example,
        if ``groups=[[0, 1, 2], [3, 4]]`` it is impossible to have a bracket
        at ``[(0, 1), (3, 4)]``...  it is only possible to do, e.g.,
        ``[0, (3, 4)]`` (single bar vs group) or  ``[(0, 1, 2), (3, 4)]``
        (full group vs full group).
      3 Bracket drawing is much better when adjacent pairs of bars are
        specified before non-adjacent pairs of bars.
    Smart defaults sets the following parameters:
        bar color: light gray (70%)
        error bar color: black
        line color: black
        bracket color: dark gray (30%)

    """
    # check matplotlib
    if plt is None:
        raise ImportError('Barplot requires matplotlib.pyplot.')
    # be nice to pandas
    if DataFrame is not None:
        if isinstance(h, DataFrame) and bar_names is None:
            if axis == 0:
                bar_names = h.columns.tolist()
            else:
                bar_names = h.index.tolist()
    # check arg errors
    if gap_size < 0 or gap_size >= 1:
        raise ValueError('Barplot argument "gap_size" must be in the range '
                         '[0, 1).')
    if err_bars is not None:
        if isinstance(err_bars, string_types) and \
                err_bars not in ['sd', 'se', 'ci']:
            raise ValueError('err_bars must be "sd", "se", or "ci" (or an '
                             'array of error bar magnitudes).')
    # handle single-element args
    if isinstance(bracket_text, string_types):
        bracket_text = [bracket_text]
    if isinstance(group_names, string_types):
        group_names = [group_names]
    # arg defaults
    if bar_kwargs is None:
        bar_kwargs = dict()
    if err_kwargs is None:
        err_kwargs = dict()
    if line_kwargs is None:
        line_kwargs = dict()
    if bracket_kwargs is None:
        bracket_kwargs = dict()
    if figure_kwargs is None:
        figure_kwargs = dict()
    # user-supplied Axes
    if ax is not None:
        bar_kwargs['axes'] = ax
    # smart defaults
    if smart_defaults:
        if 'color' not in bar_kwargs.keys():
            bar_kwargs['color'] = '0.7'
        if 'color' not in line_kwargs.keys():
            line_kwargs['color'] = 'k'
        if 'ecolor' not in err_kwargs.keys():
            err_kwargs['ecolor'] = 'k'
        if 'color' not in bracket_kwargs.keys():
            bracket_kwargs['color'] = '0.3'
    # parse heights
    h = np.array(h)
    if len(h.shape) > 2:
        raise ValueError('Barplot "h" must have 2 or fewer dimensions.')
    elif len(h.shape) < 2:
        heights = np.atleast_1d(h)
    else:
        heights = h.mean(axis=axis)
    # grouping
    num_bars = len(heights)
    if groups is None:
        groups = [[x] for x in range(num_bars)]
    groups = [list(x) for x in groups]  # forgive list/tuple mix-ups
    num_groups = len(groups)
    if eq_group_widths:
        group_widths = [1. - gap_size for _ in range(num_groups)]
        group_edges = [x + gap_size / 2. for x in range(num_groups)]
        bar_widths = [[(1. - gap_size) / len(x) for _ in enumerate(x)]
                      for x in groups]
        bar_edges = [[gap_size / 2. + grp + (1. - gap_size) * bar / len(x) for
                      bar, _ in enumerate(x)] for grp, x in enumerate(groups)]
    else:
        bar_widths = [[1. - gap_size for _ in x] for x in groups]
        bar_edges = [[gap_size / 2. + grp * gap_size + (1. - gap_size) * bar
                      for bar in x] for grp, x in enumerate(groups)]
        group_widths = [np.sum(x) for x in bar_widths]
        group_edges = [x[0] for x in bar_edges]
    bar_edges = list(chain.from_iterable(bar_edges))
    bar_widths = list(chain.from_iterable(bar_widths))
    bar_centers = np.array(bar_edges) + np.array(bar_widths) / 2.
    group_centers = np.array(group_edges) + np.array(group_widths) / 2.
    # calculate error bars
    err = np.zeros(num_bars)  # default if no err_bars
    if err_bars is not None:
        if len(h.shape) == 2:
            if err_bars == 'sd':  # sample standard deviation
                err = h.std(axis)
            elif err_bars == 'se':  # standard error
                err = h.std(axis) / np.sqrt(h.shape[axis])
            else:  # 95% conf int
                err = 1.96 * h.std(axis) / np.sqrt(h.shape[axis])
        else:  # len(h.shape) == 1
            if isinstance(err_bars, string_types):
                raise ValueError('string arguments to "err_bars" ignored when '
                                 '"h" has fewer than 2 dimensions.')
            elif not h.shape == np.array(err_bars).shape:
                raise ValueError('When "err_bars" is array-like it must have '
                                 'the same shape as "h".')
            err = np.atleast_1d(err_bars)
        bar_kwargs['yerr'] = err
    # plot (bars and error bars)
    if ax is None:
        plt.figure(**figure_kwargs)
        p = plt.subplot(1, 1, 1)
    else:
        p = ax
    b = p.bar(bar_edges, heights, bar_widths, error_kw=err_kwargs,
              **bar_kwargs)
    # plot within-subject lines
    if lines:
        if axis == 0:
            xy = [(bar_centers, hts) for hts in h]
        else:
            xy = [(bar_centers, hts) for hts in h.T]
        for subj in xy:
            p.plot(subj[0], subj[1], **line_kwargs)
    # draw significance brackets
    if brackets is not None:
        brackets = [tuple(x) for x in brackets]  # forgive list/tuple mix-ups
        if not len(brackets) == len(bracket_text):
            raise ValueError('Mismatch between number of brackets and bracket '
                             'labels.')
        brk_offset = np.diff(p.get_ylim()) * 0.025
        brk_height = np.diff(p.get_ylim()) * 0.05
        # prelim: calculate text height
        t = plt.text(0.5, 0.5, bracket_text[0])
        t.set_bbox(dict(boxstyle='round, pad=0'))
        plt.draw()
        bb = t.get_bbox_patch().get_window_extent()
        txth = np.diff(p.transData.inverted().transform(bb),
                       axis=0).ravel()[-1]  # + brk_offset / 2.
        t.remove()
        # find highest points
        if lines and len(h.shape) == 2:  # brackets must be above lines
            apex = np.max(np.r_[np.atleast_2d(heights + err),
                                np.atleast_2d(np.max(h, axis))], axis=0)
        else:
            apex = np.atleast_1d(heights + err)
        apex = np.maximum(apex, 0)  # for negative-going bars
        gr_apex = np.array([np.max(apex[x]) for x in groups])
        # calculate bracket coords
        brk_lrx = []
        brk_lry = []
        brk_top = []
        brk_txt = []
        for pair in brackets:
            lr = []  # x
            ll = []  # y lower
            hh = []  # y upper
            ed = 0
            for br in pair:
                if hasattr(br, 'append'):  # group
                    bri = groups.index(br)
                    curc = [bar_centers[x] for x in groups[bri]]
                    curx = group_centers[bri]
                    cury = float(gr_apex[bri] + brk_offset)
                else:  # single bar
                    curc = []
                    curx = bar_centers[br]
                    cury = float(apex[br] + brk_offset)
                # adjust as necessary to avoid overlap
                allx = np.array(brk_lrx).ravel().tolist()
                if curx in brk_txt:
                    count = brk_txt.count(curx)
                    mustclear = brk_top[brk_txt.index(curx)] + \
                        count * (txth + brk_offset) - brk_offset
                    for x in curc:
                        ix = len(allx) - allx[::-1].index(x) - 1
                        mustclear = max(mustclear, brk_top[ix // 2])
                    cury = mustclear + brk_offset
                elif curx in allx:
                    #count = allx.count(curx)
                    ix = len(allx) - allx[::-1].index(curx) - 1
                    cury = brk_top[ix // 2] + brk_offset  # * count
                for l, r in brk_lrx:
                    if l < curx < r and cury < max(brk_top):
                        ed += 1
                # draw horiz line spanning groups if desired
                if hasattr(br, 'append') and bracket_group_lines:
                    gbr = [bar_centers[x] for x in groups[bri]]
                    gbr = (min(gbr), max(gbr))
                    p.plot(gbr, (cury, cury), **bracket_kwargs)
                # store adjusted values
                lr.append(curx)
                ll.append(cury)
                hh.append(cury + brk_height + ed * txth)
            brk_lrx.append(tuple(lr))
            brk_lry.append(tuple(ll))
            brk_top.append(np.max(hh))
            brk_txt.append(np.mean(lr))  # text x
        # plot brackets
        for ((xl, xr), (yl, yr), yh, tx, st) in zip(brk_lrx, brk_lry, brk_top,
                                                    brk_txt, bracket_text):
            # bracket lines
            lline = ((xl, xl), (yl, yh))
            rline = ((xr, xr), (yr, yh))
            hline = ((xl, xr), (yh, yh))
            for x, y in [lline, rline, hline]:
                p.plot(x, y, **bracket_kwargs)
            # bracket text
            txt = p.annotate(st, (tx, yh), xytext=(0, 2),
                             textcoords='offset points', ha='center',
                             va='baseline', annotation_clip=False)
            txt.set_bbox(dict(facecolor='w', alpha=0, boxstyle='round, pad=1'))
            # boost ymax if needed
            ybnd = p.get_ybound()
            if ybnd[-1] < yh + txth:
                p.set_ybound(ybnd[0], yh + txth)
    # annotation
    box_off(p)
    p.tick_params(axis='x', length=0, pad=12)
    p.xaxis.set_ticks(bar_centers)
    if bar_names is not None:
        p.xaxis.set_ticklabels(bar_names, va='baseline')
    if group_names is not None:
        ymin = ylim[0] if ylim is not None else p.get_ylim()[0]
        yoffset = -2 * rcParams['font.size']
        for gn, gp in zip(group_names, group_centers):
            p.annotate(gn, xy=(gp, ymin), xytext=(0, yoffset),
                       xycoords='data', textcoords='offset points',
                       ha='center', va='baseline')
    # axis limits
    p.set_xlim(0, bar_edges[-1] + bar_widths[-1] + gap_size / 2)
    if ylim is not None:
        p.set_ylim(ylim)
    # output file
    if fname is not None:
        from os.path import splitext
        fmt = splitext(fname)[-1][1:]
        plt.savefig(fname, format=fmt, transparent=True)
    # return handles for subplot and barplot instances
    plt.draw()
    return (p, b)


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
