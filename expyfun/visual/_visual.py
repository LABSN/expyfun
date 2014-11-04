"""Tools for drawing shapes and text on the screen"""

# Authors: Dan McCloy <drmccloy@uw.edu>
#          Eric Larson <larsoner@uw.edu>
#
# License: BSD (3-clause)

# make RawImage work

import numpy as np
from matplotlib.colors import colorConverter

from .._utils import check_units, string_types


def _convert_color(color):
    """Convert 3- or 4-element color into OpenGL usable color"""
    color = (0., 0., 0., 0.) if color is None else color
    color = 255 * np.array(colorConverter.to_rgba(color))
    color = color.astype(np.uint8)
    return tuple(color)


def _replicate_color(color, pts):
    """Convert single color to color array for OpenGL trianglulations"""
    return np.tile(color, len(pts) // 2)


##############################################################################
# Text

class Text(object):
    """A text object

    Parameters
    ----------
    ec : instance of ExperimentController
        Parent EC.
    text : str
        The text to display.
    pos : array
        2-element array consisting of X- and Y-position coordinates.
    color : matplotlib Color
        Color of the text.
    font_name : str
        Font to use.
    font_size : float
        Font size (points) to use.
    height : float | None
        Height of the text region. None will automatically allocate the
        necessary size.
    width : float | None | str
        Width (in pixels) of the text region. `'auto'` will allocate 80% of
        the screen width, useful for instructions. None will automatically
        allocate sufficient space, but not that this disables text wrapping.
    anchor_x : str
        Horizontal text anchor (e.g., ``'center'``).
    anchor_y : str
        Vertical text anchor (e.g., ``'center'``).
    units : str
        Units to use. These will apply to all spatial aspects of the drawing.
        shape e.g. size, position. See ``check_units`` for options.
    wrap : bool
        Whether or not the text will wrap to fit in screen, appropriate for
        multiline text. Inappropriate for text requiring precise positioning.

    Returns
    -------
    text : instance of Text
        The text object.
    """
    def __init__(self, ec, text, pos=(0, 0), color='white',
                 font_name='Arial', font_size=24, height=None,
                 width='auto', anchor_x='center', anchor_y='center',
                 units='norm', wrap=False):
        import pyglet
        pos = np.array(pos)[:, np.newaxis]
        pos = ec._convert_units(pos, units, 'pix')
        if width == 'auto':
            width = float(ec.window_size_pix[0]) * 0.8
        elif isinstance(width, string_types):
            raise ValueError('"width", if str, must be "auto"')
        self._text = pyglet.text.Label(text + ' ', x=pos[0], y=pos[1],
                                       width=width, height=height,
                                       multiline=wrap, dpi=int(ec.dpi),
                                       anchor_x=anchor_x,
                                       anchor_y=anchor_y)
        self._text.color = _convert_color(color)
        self._text.font_name = font_name
        self._text.font_size = font_size

    def set_color(self, color):
        """Set the text color

        Parameters
        ----------
        color : matplotlib Color | None
            The color. Use None for no color.
        """
        self._text.color = _convert_color(color)

    def draw(self):
        """Draw the object to the display buffer"""
        self._text.draw()


##############################################################################
# Triangulations

class _Triangular(object):
    """Super class for objects that use trianglulations and/or lines"""
    def __init__(self, ec, fill_color, line_color, line_width, line_loop):
        self._ec = ec
        self.set_fill_color(fill_color)
        self.set_line_color(line_color)
        self._line_width = line_width
        self._line_loop = line_loop  # whether or not lines drawn are looped

    def set_fill_color(self, fill_color):
        """Set the object color

        Parameters
        ----------
        fill_color : matplotlib Color | None
            The fill color. Use None for no fill.
        """
        self._fill_color = _convert_color(fill_color)

    def set_line_color(self, line_color):
        """Set the object color

        Parameters
        ----------
        fill_color : matplotlib Color | None
            The fill color. Use None for no fill.
        """
        self._line_color = _convert_color(line_color)

    def set_line_width(self, line_width):
        """Set the line width in pixels

        Parameters
        ----------
        line_width : float
            The line width. Must be given in pixels. Due to OpenGL
            limitations, it must be `0.0 <= line_width <= 10.0`.
        """
        line_width = float(line_width)
        if not (0.0 <= line_width <= 10.0):
            raise ValueError('line_width must be between 0 and 10')
        self._line_width = line_width

    def draw(self):
        """Draw the object to the display buffer"""
        import pyglet
        from pyglet import gl
        if self._points is not None and self._fill_color is not None:
            color = _replicate_color(self._fill_color, self._points)
            pyglet.graphics.draw_indexed(len(self._points) // 2,
                                         gl.GL_TRIANGLES,
                                         self._tris,
                                         ('v2f', self._points),
                                         ('c4B', color))
        if (self._line_points is not None and self._line_width > 0.0 and
                self._line_color is not None):
            color = _replicate_color(self._line_color, self._line_points)
            gl.glLineWidth(self._line_width)
            if self._line_loop:
                gl_cmd = gl.GL_LINE_LOOP
            else:
                gl_cmd = gl.GL_LINE_STRIP
            pyglet.graphics.draw(len(self._line_points) // 2,
                                 gl_cmd,
                                 ('v2f', self._line_points),
                                 ('c4B', color))


class Line(_Triangular):
    """A connected set of line segments

    Parameters
    ----------
    ec : instance of ExperimentController
        Parent EC.
    coords : array-like
        2 x N set of X, Y coordinates.
    units : str
        Units to use. These will apply to all spatial aspects of the drawing.
        shape e.g. size, position. See ``check_units`` for options.
    line_color : matplotlib Color
        Color of the line.
    line_width : float
        Line width in pixels.
    line_loop : bool
        If True, the last point will be joined to the first in a loop.

    Returns
    -------
    line : instance of Line
        The line object.
    """
    def __init__(self, ec, coords, units='norm', line_color='white',
                 line_width=1.0, line_loop=False):
        _Triangular.__init__(self, ec, fill_color=None, line_color=line_color,
                             line_width=line_width, line_loop=line_loop)
        self._points = None
        self._tris = None
        self.set_coords(coords, units)
        self.set_line_color(line_color)

    def set_coords(self, coords, units='norm'):
        """Set line coordinates

        Parameters
        ----------
        coords : array-like
            2 x N set of X, Y coordinates.
        """
        check_units(units)
        coords = np.array(coords, dtype=float)
        if coords.ndim == 1:
            coords = coords[:, np.newaxis]
        if coords.ndim != 2 or coords.shape[0] != 2:
            raise ValueError('coords must be a vector of length 2, or an '
                             'array with 2 dimensions (with first dimension '
                             'having length 2')
        coords = self._ec._convert_units(coords, units, 'pix')
        self._line_points = coords.T.flatten()


class Triangle(_Triangular):
    """A triangle

    Parameters
    ----------
    ec : instance of ExperimentController
        Parent EC.
    coords : array-like
        2 x 3 set of X, Y coordinates.
    units : str
        Units to use. These will apply to all spatial aspects of the drawing.
        shape e.g. size, position. See ``check_units`` for options.
    fill_color : matplotlib Color
        Color of the triangle.
    line_color : matplotlib Color | None
        Color of the border line. None is transparent.
    line_width : float
        Line width in pixels.

    Returns
    -------
    line : instance of Triangle
        The triangle object.
    """
    def __init__(self, ec, coords, units='norm', fill_color='white',
                 line_color=None, line_width=1.0):
        _Triangular.__init__(self, ec, fill_color=fill_color,
                             line_color=line_color, line_width=line_width,
                             line_loop=True)
        self.set_coords(coords, units)
        self.set_fill_color(fill_color)

    def set_coords(self, coords, units='norm'):
        """Set triangle coordinates

        Parameters
        ----------
        coords : array-like
            2 x 3 set of X, Y coordinates.
        """
        check_units(units)
        coords = np.array(coords, dtype=float)
        if coords.shape != (2, 3):
            raise ValueError('coords must be an array of size 2 x 3')
        coords = self._ec._convert_units(coords, units, 'pix')
        self._points = coords.T.flatten()
        self._tris = np.array([0, 1, 2])
        self._line_points = self._points


class Rectangle(_Triangular):
    """A rectangle

    Parameters
    ----------
    ec : instance of ExperimentController
        Parent EC.
    pos : array-like
        4-element array-like with X, Y center and width, height.
    units : str
        Units to use. These will apply to all spatial aspects of the drawing.
        shape e.g. size, position. See ``check_units`` for options.
    fill_color : matplotlib Color | None
        Color to fill with. None is transparent.
    line_color : matplotlib Color | None
        Color of the border line. None is transparent.
    line_width : float
        Line width in pixels.

    Returns
    -------
    line : instance of Rectangle
        The rectangle object.
    """
    def __init__(self, ec, pos, units='norm', fill_color='white',
                 line_color=None, line_width=1.0):
        _Triangular.__init__(self, ec, fill_color=fill_color,
                             line_color=line_color, line_width=line_width,
                             line_loop=True)
        self.set_pos(pos, units)

    def set_pos(self, pos, units='norm'):
        """Set the position of the rectangle

        Parameters
        ----------
        pos : array-like
            X, Y, width, height of the rectangle.
        units : str
            Units to use. See ``check_units`` for options.
        """
        check_units(units)
        # do this in normalized units, then convert
        pos = np.array(pos)
        if not (pos.ndim == 1 and pos.size == 4):
            raise ValueError('pos must be a 4-element array-like vector')
        self._pos = pos
        w = self._pos[2]
        h = self._pos[3]
        points = np.array([[-w / 2., -h / 2.],
                           [-w / 2., h / 2.],
                           [w / 2., h / 2.],
                           [w / 2., -h / 2.]]).T
        points += np.array(self._pos[:2])[:, np.newaxis]
        points = self._ec._convert_units(points, units, 'pix')
        self._points = points.T.flatten()
        self._tris = np.array([0, 1, 2, 0, 2, 3])
        self._line_points = self._points  # all 4 points used for line drawing


class Diamond(_Triangular):
    """A diamond

    Parameters
    ----------
    ec : instance of ExperimentController
        Parent EC.
    pos : array-like
        4-element array-like with X, Y center and width, height.
    units : str
        Units to use. These will apply to all spatial aspects of the drawing.
        shape e.g. size, position. See ``check_units`` for options.
    fill_color : matplotlib Color | None
        Color to fill with. None is transparent.
    line_color : matplotlib Color | None
        Color of the border line. None is transparent.
    line_width : float
        Line width in pixels.

    Returns
    -------
    line : instance of Rectangle
        The rectangle object.
    """
    def __init__(self, ec, pos, units='norm', fill_color='white',
                 line_color=None, line_width=1.0):
        _Triangular.__init__(self, ec, fill_color=fill_color,
                             line_color=line_color, line_width=line_width,
                             line_loop=True)
        self.set_pos(pos, units)

    def set_pos(self, pos, units='norm'):
        """Set the position of the rectangle

        Parameters
        ----------
        pos : array-like
            X, Y, width, height of the rectangle.
        units : str
            Units to use. See ``check_units`` for options.
        """
        check_units(units)
        # do this in normalized units, then convert
        pos = np.array(pos)
        if not (pos.ndim == 1 and pos.size == 4):
            raise ValueError('pos must be a 4-element array-like vector')
        self._pos = pos
        w = self._pos[2]
        h = self._pos[3]
        points = np.array([[w / 2., 0.],
                           [0., h / 2.],
                           [-w / 2., 0.],
                           [0., -h / 2.]]).T
        points += np.array(self._pos[:2])[:, np.newaxis]
        points = self._ec._convert_units(points, units, 'pix')
        self._points = points.T.flatten()
        self._tris = np.array([0, 1, 2, 0, 2, 3])
        self._line_points = self._points  # all 4 points used for line drawing


class Circle(_Triangular):
    """A circle or ellipse

    Parameters
    ----------
    ec : instance of ExperimentController
        Parent EC.
    radius : float | array-like
        Radius of the circle. Can be array-like with two elements to
        make an ellipse.
    pos : array-like
        2-element array-like with X, Y center positions.
    units : str
        Units to use. These will apply to all spatial aspects of the drawing.
        shape e.g. size, position. See ``check_units`` for options.
    n_edges : int
        Number of edges to use (must be >= 4) to approximate a circle.
    fill_color : matplotlib Color | None
        Color to fill with. None is transparent.
    line_color : matplotlib Color | None
        Color of the border line. None is transparent.
    line_width : float
        Line width in pixels.

    Returns
    -------
    circle : instance of Circle
        The circle object.
    """
    def __init__(self, ec, radius=1, pos=(0, 0), units='norm',
                 n_edges=200, fill_color='white', line_color=None,
                 line_width=1.0):
        _Triangular.__init__(self, ec, fill_color=fill_color,
                             line_color=line_color, line_width=line_width,
                             line_loop=True)
        if not isinstance(n_edges, int):
            raise TypeError('n_edges must be an int')
        if n_edges < 4:
            raise ValueError('n_edges must be >= 4 for a reasonable circle')
        self._n_edges = n_edges

        # need to set a dummy value here so recalculation doesn't fail
        self._radius = np.array([1., 1.])
        self.set_pos(pos, units)
        self.set_radius(radius, units)

        # construct triangulation (never changes so long as n_edges is fixed)
        tris = [[0, ii + 1, ii + 2] for ii in range(n_edges)]
        tris = np.concatenate(tris)
        tris[-1] = 1  # fix wrap for last triangle
        self._tris = tris

    def set_radius(self, radius, units='norm'):
        """Set the position and radius of the circle

        Parameters
        ----------
        radius : array-like | float
            X- and Y-direction extents (radii) of the circle / ellipse.
            A single value (float) will be replicated for both directions.
        units : str
            Units to use. See ``check_units`` for options.
        """
        check_units(units)
        radius = np.atleast_1d(radius).astype(float)
        if radius.ndim != 1 or radius.size > 2:
            raise ValueError('radius must be a 1- or 2-element '
                             'array-like vector')
        if radius.size == 1:
            radius = np.r_[radius, radius]
        # convert to pixel (OpenGL) units
        self._radius = self._ec._convert_units(radius[:, np.newaxis],
                                               units, 'pix')[:, 0]
        # need to subtract center position
        ctr = self._ec._convert_units(np.zeros((2, 1)), units, 'pix')[:, 0]
        self._radius -= ctr
        self._recalculate()

    def set_pos(self, pos, units='norm'):
        """Set the position and radius of the circle

        Parameters
        ----------
        pos : array-like
            X, Y center of the circle.
        units : str
            Units to use. See ``check_units`` for options.
        """
        check_units(units)
        pos = np.array(pos, dtype=float)
        if not (pos.ndim == 1 and pos.size == 2):
            raise ValueError('pos must be a 2-element array-like vector')
        # convert to pixel (OpenGL) units
        self._pos = self._ec._convert_units(pos[:, np.newaxis],
                                            units, 'pix')[:, 0]
        self._recalculate()

    def _recalculate(self):
        """Helper to recalculate point coordinates"""
        edges = self._n_edges
        arg = 2 * np.pi * (np.arange(edges) / float(edges))
        points = np.array([self._radius[0] * np.cos(arg),
                           self._radius[1] * np.sin(arg)])
        points = np.c_[np.zeros((2, 1)), points]  # prepend the center
        points += np.array(self._pos[:2], dtype=float)[:, np.newaxis]
        self._points = points.T.ravel()
        self._line_points = self._points[2:]  # omit center point for lines


class ConcentricCircles(object):
    """A set of filled concentric circles drawn without edges

    Parameters
    ----------
    ec : instance of ExperimentController
        Parent EC.
    radii : list of float
        Radii of the circles. Note that circles will be drawn in order,
        so using e.g., radii=[1., 2.] will cause the first circle to be
        covered by the second.
    pos : array-like
        2-element array-like with the X, Y center position.
    units : str
        Units to use. These will apply to all spatial aspects of the drawing.
        See ``check_units`` for options.
    colors : list or tuple of matplotlib Colors
        Color to fill each circle with.

    Returns
    -------
    circle : instance of Circle
        The circle object.
    """
    def __init__(self, ec, radii=(0.2, 0.05), pos=(0, 0), units='norm',
                 colors=('w', 'k')):
        radii = np.array(radii, float)
        if radii.ndim != 1:
            raise ValueError('radii must be 1D')
        if not isinstance(colors, (tuple, list)):
            raise TypeError('colors must be a tuple, list, or array')
        if len(colors) != len(radii):
            raise ValueError('colors and radii must be the same length')
        # need to set a dummy value here so recalculation doesn't fail
        self._circles = [Circle(ec, r, pos, units, fill_color=c, line_width=0)
                         for r, c in zip(radii, colors)]

    def __len__(self):
        return len(self._circles)

    def set_pos(self, pos, units='norm'):
        """Set the position of the circles

        Parameters
        ----------
        pos : array-like
            X, Y center of the circle.
        units : str
            Units to use. See ``check_units`` for options.
        """
        for circle in self._circles:
            circle.set_pos(pos, units)

    def set_radius(self, radius, idx, units='norm'):
        """Set the radius of one of the circles

        Parameters
        ----------
        radius : float
            Radius the circle.
        idx : int
            Index of the circle.
        units : str
            Units to use. See ``check_units`` for options.
        """
        self._circles[idx].set_radius(radius, units)

    def set_radii(self, radii, units='norm'):
        """Set the color of each circle

        Parameters
        ----------
        radii : array-like
            List of radii to assign to the circles. Must contain the same
            number of radii as the number of circles.
        units : str
            Units to use. See ``check_units`` for options.
        """
        radii = np.array(radii, float)
        if radii.ndim != 1 or radii.size != len(self):
            raise ValueError('radii must contain exactly {0} radii'
                             ''.format(len(self)))
        for idx, radius in enumerate(radii):
            self.set_radius(radius, idx, units)

    def set_color(self, color, idx):
        """Set the color of one of the circles

        Parameters
        ----------
        color : matplotlib Color
            Color of the circle.
        idx : int
            Index of the circle.
        units : str
            Units to use. See ``check_units`` for options.
        """
        self._circles[idx].set_fill_color(color)

    def set_colors(self, colors):
        """Set the color of each circle

        Parameters
        ----------
        colors : list or tuple of matplotlib Colors
            Must be of type list or tuple, and contain the same number of
            colors as the number of circles.
        """
        if not isinstance(colors, (tuple, list)) or len(colors) != len(self):
            raise ValueError('colors must be a list or tuple with {0} colors'
                             ''.format(len(self)))
        for idx, color in enumerate(colors):
            self.set_color(color, idx)

    def draw(self):
        """Draw the fixation dot
        """
        for circle in self._circles:
            circle.draw()


class FixationDot(ConcentricCircles):
    """A reasonable centered fixation dot

    This uses concentric circles, the inner of which has a radius of one
    pixel, to create a fixation dot. If finer-grained control is desired,
    consider using ``ConcentricCircles``.

    Parameters
    ----------
    ec : instance of ExperimentController
        Parent EC.
    colors : list of matplotlib Colors
        Color to fill the outer and inner circle with, respectively.

    Returns
    -------
    fix : instance of FixationDot
        The fixation dot.
    """
    def __init__(self, ec, colors=('w', 'k')):
        if len(colors) != 2:
            raise ValueError('colors must have length 2')
        super(FixationDot, self).__init__(ec, radii=[0.2, 0.2],
                                          pos=[0, 0], units='deg',
                                          colors=colors)
        self.set_radius(1, 1, units='pix')


##############################################################################
# Image display

class RawImage(object):
    """Create image from array for on-screen display

    Parameters
    ----------
    ec : instance of ExperimentController
        Parent EC.
    image_buffer : array
        N x M x 3 (or 4) array. Color values should range between 0 and 1.
    pos : array-like
        2-element array-like with X, Y (center) arguments.
    scale : float
        The scale factor. 1 is native size (pixel-to-pixel), 2 is twice as
        large, etc.
    units : str
        Units to use for the position. See ``check_units`` for options.

    Returns
    -------
    img : instance of RawImage
        The image object.
    """
    def __init__(self, ec, image_buffer, pos=(0, 0), scale=1., units='norm'):
        self._ec = ec
        self._img = None
        self.set_image(image_buffer)
        self.set_pos(pos, units)
        self.set_scale(scale)

    def set_image(self, image_buffer):
        """Set image buffer data

        Parameters
        ----------
        image_buffer : array
            N x M x 3 (or 4) array. Can be type ``np.float64`` or ``np.uint8``.
            If ``np.float64``, color values must range between 0 and 1.
            ``np.uint8`` is slightly more efficient.
        """
        from pyglet import image, sprite
        image_buffer = np.ascontiguousarray(image_buffer)
        if image_buffer.dtype not in (np.float64, np.uint8):
            raise TypeError('image_buffer must be np.float64 or np.uint8')
        if image_buffer.dtype == np.float64:
            if image_buffer.max() > 1 or image_buffer.min() < 0:
                raise ValueError('all float values must be between 0 and 1')
            image_buffer = (image_buffer * 255).astype('uint8')
        if not image_buffer.ndim == 3 or image_buffer.shape[2] not in [3, 4]:
            raise RuntimeError('image_buffer incorrect size: {}'
                               ''.format(image_buffer.shape))
        # add alpha channel if necessary
        dims = image_buffer.shape
        fmt = 'RGB' if dims[2] == 3 else 'RGBA'
        self._sprite = sprite.Sprite(image.ImageData(dims[1], dims[0], fmt,
                                                     image_buffer.tostring(),
                                                     -dims[1] * dims[2]))

    def set_pos(self, pos, units='norm'):
        """Set image position

        Parameters
        ----------
        ec : instance of ExperimentController
            Parent EC.
        pos : array-like
            2-element array-like with X, Y (center) arguments.
        units : str
            Units to use. See ``check_units`` for options.
        """
        pos = np.array(pos, float)
        if pos.ndim != 1 or pos.size != 2:
            raise ValueError('pos must be a 2-element array')
        pos = np.reshape(pos, (2, 1))
        self._pos = self._ec._convert_units(pos, units, 'pix').ravel()

    @property
    def bounds(self):
        """L, B, W, H (in pixels) of the image"""
        pos = np.array(self._pos, float)
        size = np.array([self._sprite.width,
                         self._sprite.height], float)
        bounds = np.concatenate((pos - size / 2., pos + size / 2.))
        return bounds[[0, 2, 1, 3]]

    @property
    def scale(self):
        return self._scale

    def set_scale(self, scale):
        """Create image from array for on-screen display

        Parameters
        ----------
        ec : instance of ExperimentController
            Parent EC.
        pos : array-like
            2-element array-like with X, Y (center) arguments.
        units : str
            Units to use. See ``check_units`` for options.
        """
        scale = float(scale)
        self._scale = scale
        self._sprite.scale = self._scale

    def draw(self):
        """Draw the image to the buffer"""
        self._sprite.scale = self._scale
        pos = self._pos - [self._sprite.width / 2., self._sprite.height / 2.]
        self._sprite.set_position(pos[0], pos[1])
        self._sprite.draw()
