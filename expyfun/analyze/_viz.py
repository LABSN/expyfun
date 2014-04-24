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
        A keyword indicating the formatting scheme. Currently supports "ross"
        and "default"; any other string will yield the same as "default".

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
        between the third and fourth bars, specify as [(0,1),(2,3)]. If you
        want brackets between groups of bars instead of between bars, indicate
        the group numbers as singleton lists within the tuple: [([0], [1])].
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
        two_d = False
    else:
        heights = h.mean(axis=axis)
        two_d = True
    # grouping
    num_bars = len(heights)
    if groups is None:
        groups = [[x] for x in range(num_bars)]
    num_groups = len(groups)
    if eq_group_widths:
        group_widths = [1 - gap_size for _ in range(num_groups)]
        group_edges = [x + gap_size for x in range(num_groups)]
        bar_widths = [[(1 - gap_size) / len(x) for _ in enumerate(x)]
                      for x in groups]
        bar_edges = [[gap_size / 2 + grp + (1 - gap_size) * bar / len(x) for
                      bar, _ in enumerate(x)] for grp, x in enumerate(groups)]
    else:
        bar_widths = [[1 - gap_size for _ in x] for x in groups]
        bar_edges = [[gap_size / 2 + grp * gap_size + (1 - gap_size) * bar for
                      bar in x] for grp, x in enumerate(groups)]
        group_widths = [np.sum(x) for x in bar_widths]
        group_edges = [x[0] for x in bar_edges]

    bar_edges = list(chain.from_iterable(bar_edges))
    bar_widths = list(chain.from_iterable(bar_widths))
    bar_centers = np.array(bar_edges) + np.array(bar_widths) / 2
    group_centers = np.array(group_edges) + np.array(group_widths) / 2
    # calculate error bars
    if err_bars is not None:
        if two_d:
            if err_bars == 'sd':  # sample standard deviation
                err = h.std(axis=axis)
            elif err_bars == 'se':  # standard error
                h.shape[axis]
                err = h.std(axis) / np.sqrt(h.shape[axis])
            else:  # 95% conf int
                err = 1.96 * h.std(axis) / np.sqrt(h.shape[axis])
        else:  # two_d == False
            if isinstance(err_bars, string_types):
                raise ValueError('string arguments to "err_bars" ignored when '
                                 '"h" has fewer than 2 dimensions.')
            else:
                err_bars = np.atleast_1d(err_bars)
                if not h.shape == err_bars.shape:
                    raise ValueError('When "err_bars" is array-like it must '
                                     'have the same shape as barplot arg "h".')
            err = err_bars
        bar_kwargs['yerr'] = err
    else:  # still must define err (for signif. brackets)
        err = np.zeros(num_bars)
    # plot (bars and error bars)
    if ax is None:
        plt.figure(**figure_kwargs)
        p = plt.subplot(1, 1, 1)
    else:
        p = ax
    b = p.bar(bar_edges, heights, bar_widths, error_kw=err_kwargs,
              **bar_kwargs)
    # within-subject lines
    if two_d:
        max_pts = np.max(h, axis)
    else:
        max_pts = heights
    if lines:
        if axis == 0:
            xy = [(bar_centers, hts) for hts in h]
        else:
            xy = [(bar_centers, hts) for hts in h.T]
        for subj in xy:
            p.plot(subj[0], subj[1], **line_kwargs)
    else:
        max_pts.fill(0)
    # significance brackets
    if brackets is not None:
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
        brk_txt_h = np.diff(p.transData.inverted().transform(bb),
                            axis=0).ravel()[-1] + brk_offset/2.
        t.remove()
        # find apices of bars
        apex = np.max(np.r_[np.atleast_2d(heights + err),
                            np.atleast_2d(max_pts)], axis=0)
        gr_apex = np.array([np.max(apex[x]) for x in groups])
        # apices of brackets
        brk_list = []  # np.zeros((len(brackets), 2, 2), dtype=float)
        brk_apex_list = []
        # calculate bracket coords
        for pair, text in zip(brackets, bracket_text):
            if len(pair) != 2:
                raise ValueError('brackets must be list of 2-element tuples.')
            ylo = []
            xlr = []
            for br in pair:
                if hasattr(br, 'append'):  # it's a group, not a single bar
                    br = br[0]
                    xlr.append(group_centers[br])
                    ylo.append(gr_apex[br] + brk_offset)
                    # horizontal line spanning group:
                    if bracket_group_lines:
                        gbr = (bar_centers[groups[br][0]],
                               bar_centers[groups[br][-1]])
                        p.plot(gbr, (ylo[-1], ylo[-1]), **bracket_kwargs)
                    # update apices to prevent bracket overlap
                    yhi = max(ylo) + brk_height
                    while np.any([np.abs(x - yhi) < brk_offset for x in
                                  brk_apex_list]):
                        yhi += (brk_txt_h + 2 * brk_offset)
                    gr_apex[br] = yhi + brk_txt_h
                else:
                    xlr.append(bar_centers[br])
                    ylo.append(apex[br] + brk_offset)
                    yhi = max(ylo) + brk_height
                    while np.any([np.abs(x - yhi) < brk_offset for x in
                                  brk_apex_list]):
                        yhi += (brk_txt_h + 2 * brk_offset)
                    # update apices to prevent bracket overlap
                    apex[br] = yhi + brk_txt_h
                    new_ga = np.array([np.max(apex[x]) for x in groups])
                    gr_apex[new_ga > gr_apex] = new_ga[new_ga > gr_apex]
            # points defining brackets
            lbr = ((xlr[0], xlr[0]), (ylo[0], yhi))
            rbr = ((xlr[1], xlr[1]), (ylo[1], yhi))
            hbr = (tuple(xlr), (yhi, yhi))
            brk_list.append([lbr, rbr, hbr])
            brk_apex_list.append(yhi)
            for x, y in [lbr, rbr, hbr]:
                p.plot(x, y, **bracket_kwargs)
            # bracket text
            txt = p.annotate(text, (np.mean(xlr), yhi + brk_offset/2.),
                             xytext=(0, 1), textcoords='offset points',
                             ha='center', annotation_clip=False)
            txt.set_bbox(dict(facecolor='w', alpha=0, boxstyle='round, pad=1'))
            plt.draw()
            txtb = txt.get_bbox_patch().get_window_extent()
            txtbb = p.transData.inverted().transform(txtb).ravel()[-1]
            ybnd = p.get_ybound()
            if txtbb > ybnd[-1]:
                p.set_ybound(ybnd[0], txtbb)
    # annotation
    box_off(p)
    p.tick_params(axis='x', length=0, pad=12)
    p.xaxis.set_ticks(bar_centers)
    if bar_names is not None:
        p.xaxis.set_ticklabels(bar_names, va='baseline')
    if group_names is not None:
        yoffset = -2 * rcParams['font.size']
        for gn, gp in zip(group_names, group_centers):
            p.annotate(gn, xy=(gp, 0), xytext=(0, yoffset),
                       textcoords='offset points', ha='center', va='baseline')
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
