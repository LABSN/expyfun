"""Analysis visualization functions
"""

import numpy as np
from itertools import chain

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
        brk_l = '{{'
        brk_r = '}}'
    else:
        wrap = ''
        brk_l = ''
        brk_r = ''
    if scheme == 'ross':  # (exact value up to 4 decimal places)
        pv[pval >= 0.0001] = [wrap + 'p = {:.4f}'.format(x) + wrap
                              for x in pval[pval > 0.0001]]
        pv[pval < 0.0001] = [wrap + 'p < 10^' + brk_l + '{}'.format(x) +
                             brk_r + wrap for x in expon[pval < 0.0001]]
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
        pv[pval < 0.0001] = [wrap + 'p < 10^' + brk_l + '{}'.format(x) +
                             brk_r + wrap for x in expon[pval < 0.0001]]
    if single_value:
        pv = pv[0]
    return(pv)


def _instantiate(obj, typ):
    """Returns obj if obj is not None, else returns new instance of typ
    obj : an object
        An object (most likely one that a user passed into a function) that,
        if ``None``, should be initiated as an empty object of some other type.
    typ : an object type
        Expected values are list, dict, int, bool, etc.
    """
    return typ() if obj is None else obj


def barplot(h, axis=-1, ylim=None, err_bars=None, lines=False,
            groups=None, eq_group_widths=False, gap_size=0.2,
            brackets=None, bracket_text=None, bracket_inline=False,
            bracket_group_lines=False, bar_names=None, group_names=None,
            bar_kwargs=None, err_kwargs=None, line_kwargs=None,
            bracket_kwargs=None, pval_kwargs=None, figure_kwargs=None,
            smart_defaults=True, fname=None, ax=None):
    """Makes barplots w/ optional line overlays, grouping, & signif. brackets.

    Parameters
    ----------
    h : array-like
        If `h` is 2-dimensional, heights will be calculated as means along
        the axis given by `axis`. If `h` is of lower dimension, it is
        treated as raw height values. If `h` is a `pandas.DataFrame` and
        `bar_names` is ``None``, `bar_names` will be inferred from the
        DataFrame's `column` labels (if ``axis=0``) or `index` labels.
    axis : int
        The axis along which to calculate mean values to determine bar heights.
        Ignored if `h` is 0- or 1-dimensional.
    ylim : tuple | None
        y-axis limits passed to `matplotlib.pyplot.subplot.set_ylim`.
    err_bars : str | array-like | None
        Type of error bars to be added to the barplot. Possible values are
        ``'sd'`` for sample standard deviation, ``'se'`` for standard error of
        the mean, or ``'ci'`` for 95% confidence interval. If ``None``, no
        error bars will be plotted. Custom error bar heights are possible by
        passing an array-like object; in such cases `err_bars` must have the
        same dimensionality and shape as `h`.
    lines : bool
        Whether to plot within-subject data as lines overlaid on the barplot.
    groups : list | None
        List of lists containing the integers in ``range(num_bars)``, with
        sub-lists indicating the desired grouping. For example, if `h` has
        has shape (10, 4) and ``axis = -1`` then "num_bars" is 4; if you want
        the first bar isolated and the remaining three grouped, then specify
        ``groups=[[0], [1, 2, 3]]``.
    eq_group_widths : bool
        Should all groups have the same width? If ``False``, all bars will have
        the same width. Ignored if `groups` is ``None``, since the bar/group
        distinction is meaningless in that case.
    gap_size : float
        Width of the gap between groups (if `eq_group_width` is ``True``) or
        between bars, expressed as a proportion [0,1) of group or bar width.
        Half the width of `gap_size` will be added between the outermost bars
        and the plot edges.
    brackets : list of tuples | None
        Location of significance brackets. Scheme is similar to the
        specification of `groups`; a bracket between the first and second bar
        and another between the third and fourth bars would be specified as
        ``brackets=[(0, 1), (2, 3)]``. Brackets between groups of bars instead
        of individual bars are specified as lists within the tuple:
        ``brackets=[([0, 1], [2, 3])]`` draws a single bracket between group
        ``[0, 1]`` and group ``[2, 3]``. For best results, pairs of adjacent
        bars should come earlier in the list than non-adjacent pairs.
    bracket_text : str | list | None
        Text to display above brackets.
    bracket_inline : bool
        If ``True``, bracket text will be vertically centered along a broken
        bracket line. If ``False``, text will be above the line.
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
    pval_kwargs : dict
        Arguments passed to ``matplotlib.pyplot.annotate()`` when drawing
        bracket labels.
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
        Path and name of output file. File type is inferred from the file
        extension of `fname` and should work for any of the types supported by
        pyplot (pdf, eps, svg, png, raw).
    ax : matplotlib.pyplot.axes | None
        A ``matplotlib.pyplot.axes`` instance.  If ``None``, a new figure with
        a single subplot will be created.

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
    from matplotlib import pyplot as plt, rcParams
    try:
        from pandas.core.frame import DataFrame
    except Exception:
        DataFrame = None

    # be nice to pandas
    if DataFrame is not None:
        if isinstance(h, DataFrame) and bar_names is None:
            bar_names = h.columns.tolist() if axis == 0 else h.index.tolist()
    # check arg errors
    if gap_size < 0 or gap_size >= 1:
        raise ValueError('Barplot argument "gap_size" must be in the range '
                         '[0, 1).')
    if err_bars is not None:
        if isinstance(err_bars, string_types) and \
                err_bars not in ['sd', 'se', 'ci']:
            raise ValueError('err_bars must be "sd", "se", or "ci" (or an '
                             'array of error bar magnitudes).')
    if brackets is not None:
        if any([len(x) != 2 for x in brackets]):
            raise ValueError('Each top-level element of brackets must have '
                             'length 2.')
        if not len(brackets) == len(bracket_text):
            raise ValueError('Mismatch between number of brackets and bracket '
                             'labels.')
    # handle single-element args
    if isinstance(bracket_text, string_types):
        bracket_text = [bracket_text]
    if isinstance(group_names, string_types):
        group_names = [group_names]
    # arg defaults: if arg is None, instantiate as given type
    brackets = _instantiate(brackets, list)
    bar_kwargs = _instantiate(bar_kwargs, dict)
    err_kwargs = _instantiate(err_kwargs, dict)
    line_kwargs = _instantiate(line_kwargs, dict)
    pval_kwargs = _instantiate(pval_kwargs, dict)
    figure_kwargs = _instantiate(figure_kwargs, dict)
    bracket_kwargs = _instantiate(bracket_kwargs, dict)
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
    heights = np.atleast_1d(h) if h.ndim < 2 else h.mean(axis=axis)
    # grouping
    num_bars = len(heights)
    if groups is None:
        groups = [[x] for x in range(num_bars)]
    groups = [list(x) for x in groups]  # forgive list/tuple mix-ups
    # calculate bar positions
    non_gap = 1 - gap_size
    offset = gap_size / 2.
    if eq_group_widths:
        group_sizes = np.array([float(len(_grp)) for _grp in groups], int)
        group_widths = [non_gap for _ in groups]
        group_edges = [offset + _ix for _ix in range(len(groups))]
        group_ixs = list(chain.from_iterable([range(x) for x in group_sizes]))
        bar_widths = np.repeat(np.array(group_widths) / group_sizes,
                               group_sizes).tolist()
        bar_edges = (np.repeat(group_edges, group_sizes) +
                     bar_widths * np.array(group_ixs)).tolist()
    else:
        bar_widths = [[non_gap for _ in _grp] for _grp in groups]
        # next line: offset + cumul. gap widths + cumul. bar widths
        bar_edges = [[offset + _ix * gap_size + _bar * non_gap
                      for _bar in _grp] for _ix, _grp in enumerate(groups)]
        group_widths = [np.sum(_width) for _width in bar_widths]
        group_edges = [_edge[0] for _edge in bar_edges]
        bar_edges = list(chain.from_iterable(bar_edges))
        bar_widths = list(chain.from_iterable(bar_widths))
    bar_centers = np.array(bar_edges) + np.array(bar_widths) / 2.
    group_centers = np.array(group_edges) + np.array(group_widths) / 2.
    # calculate error bars
    err = np.zeros(num_bars)  # default if no err_bars
    if err_bars is not None:
        if h.ndim == 2:
            if err_bars == 'sd':  # sample standard deviation
                err = h.std(axis)
            elif err_bars == 'se':  # standard error
                err = h.std(axis) / np.sqrt(h.shape[axis])
            else:  # 95% conf int
                err = 1.96 * h.std(axis) / np.sqrt(h.shape[axis])
        else:  # h.ndim == 1
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
        p = plt.subplot(111)
    else:
        p = ax
    b = p.bar(bar_edges, heights, bar_widths, error_kw=err_kwargs,
              **bar_kwargs)
    # plot within-subject lines
    if lines:
        _h = h if axis == 0 else h.T
        xy = [(bar_centers, hts) for hts in _h]
        for subj in xy:
            p.plot(subj[0], subj[1], **line_kwargs)
    # draw significance brackets
    if len(brackets):
        brackets = [tuple(x) for x in brackets]  # forgive list/tuple mix-ups
        brk_offset = np.diff(p.get_ylim()) * 0.025
        brk_min_h = np.diff(p.get_ylim()) * 0.05
        # temporarily plot a textbox to get its height
        t = plt.annotate(bracket_text[0], (0, 0), **pval_kwargs)
        t.set_bbox(dict(boxstyle='round, pad=0.25'))
        plt.draw()
        bb = t.get_bbox_patch().get_window_extent()
        txth = np.diff(p.transData.inverted().transform(bb),
                       axis=0).ravel()[-1]
        if bracket_inline:
            txth = txth / 2.
        t.remove()
        # find highest points
        if lines and h.ndim == 2:  # brackets must be above lines & error bars
            apex = np.amax(np.r_[np.atleast_2d(heights + err),
                                 np.atleast_2d(np.amax(h, axis))], axis=0)
        else:
            apex = np.atleast_1d(heights + err)
        apex = np.maximum(apex, 0)  # for negative-going bars
        apex = apex + brk_offset
        gr_apex = np.array([np.amax(apex[_g]) for _g in groups])
        # boolean for whether each half of a bracket is a group
        is_group = [[hasattr(_b, 'append') for _b in _br] for _br in brackets]
        # bracket left & right coords
        brk_lr = [[group_centers[groups.index(_ix)] if _g
                   else bar_centers[_ix] for _ix, _g in zip(_brk, _isg)]
                  for _brk, _isg in zip(brackets, is_group)]
        # bracket L/R midpoints (label position)
        brk_c = [np.mean(_lr) for _lr in brk_lr]
        # bracket bottom coords (first pass)
        brk_b = [[gr_apex[groups.index(_ix)] if _g else apex[_ix]
                  for _ix, _g in zip(_brk, _isg)]
                 for _brk, _isg in zip(brackets, is_group)]
        # main bracket positioning loop
        brk_t = []
        for _ix, (_brk, _isg) in enumerate(zip(brackets, is_group)):
            # which bars does this bracket span?
            spanned_bars = list(chain.from_iterable(
                [_b if hasattr(_b, 'append') else [_b] for _b in _brk]))
            spanned_bars = range(min(spanned_bars), max(spanned_bars) + 1)
            # raise apex a bit extra if prev bracket label centered on bar
            prev_label_pos = brk_c[_ix - 1] if _ix else -1
            label_bar_ix = np.where(np.isclose(bar_centers, prev_label_pos))[0]
            if any(np.array_equal(label_bar_ix, x) for x in _brk):
                apex[label_bar_ix] += txth
            elif any(_isg):
                label_bar_less = np.where(bar_centers < prev_label_pos)[0]
                label_bar_more = np.where(bar_centers > prev_label_pos)[0]
                if len(label_bar_less) and len(label_bar_more):
                    apex[label_bar_less] += txth
                    apex[label_bar_more] += txth
            gr_apex = np.array([np.amax(apex[_g]) for _g in groups])
            # recalc lower tips of bracket: apex / gr_apex may have changed
            brk_b[_ix] = [gr_apex[groups.index(_b)] if _g else apex[_b]
                          for _b, _g in zip(_brk, _isg)]
            # calculate top span position
            _min_t = max(apex[spanned_bars]) + brk_min_h
            brk_t.append(_min_t)
            # raise apex on spanned bars to account for bracket
            apex[spanned_bars] = np.maximum(apex[spanned_bars],
                                            _min_t) + brk_offset
            gr_apex = np.array([np.amax(apex[_g]) for _g in groups])
        # draw horz line spanning groups if desired
        if bracket_group_lines:
            for _brk, _isg, _blr in zip(brackets, is_group, brk_b):
                for _bk, _g, _b in zip(_brk, _isg, _blr):
                    if _g:
                        _lr = [bar_centers[_ix]
                               for _ix in groups[groups.index(_bk)]]
                        _lr = (min(_lr), max(_lr))
                        p.plot(_lr, (_b, _b), **bracket_kwargs)
        # draw (left, right, bottom-left, bottom-right, top, center, string)
        for ((_l, _r), (_bl, _br), _t, _c, _s) in zip(brk_lr, brk_b, brk_t,
                                                      brk_c, bracket_text):
            # bracket text
            defaults = dict(ha='center', annotation_clip=False,
                            textcoords='offset points')
            for k, v in defaults.items():
                if k not in pval_kwargs.keys():
                    pval_kwargs[k] = v
            if 'va' not in pval_kwargs.keys():
                pval_kwargs['va'] = 'center' if bracket_inline else 'baseline'
            if 'xytext' not in pval_kwargs.keys():
                pval_kwargs['xytext'] = (0, 0) if bracket_inline else (0, 2)
            txt = p.annotate(_s, (_c, _t), **pval_kwargs)
            txt.set_bbox(dict(facecolor='w', alpha=0,
                              boxstyle='round, pad=0.2'))
            plt.draw()
            # bracket lines
            lline = ((_l, _l), (_bl, _t))
            rline = ((_r, _r), (_br, _t))
            tline = ((_l, _r), (_t, _t))
            if bracket_inline:
                bb = txt.get_bbox_patch().get_window_extent()
                txtw = np.diff(p.transData.inverted().transform(bb),
                               axis=0).ravel()[0]
                _m = _c - txtw / 2.
                _n = _c + txtw / 2.
                tline = [((_l, _m), (_t, _t)), ((_n, _r), (_t, _t))]
            else:
                tline = [((_l, _r), (_t, _t))]
            for x, y in [lline, rline] + tline:
                p.plot(x, y, **bracket_kwargs)
            # boost ymax if needed
            ybnd = p.get_ybound()
            if ybnd[-1] < _t + txth:
                p.set_ybound(ybnd[0], _t + txth)
    # annotation
    box_off(p)
    p.tick_params(axis='x', length=0, pad=12)
    p.xaxis.set_ticks(bar_centers)
    if bar_names is not None:
        p.xaxis.set_ticklabels(bar_names, va='baseline')
    if group_names is not None:
        ymin = ylim[0] if ylim is not None else p.get_ylim()[0]
        yoffset = -2.5 * rcParams['font.size']
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

    Parameters
    ----------
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

    Returns
    -------
    ax : matplotlib Axes
        The axes used to plot the image.
    """
    import matplotlib.pyplot as plt
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
