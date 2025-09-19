"""Visual stimulus design.

Tools for drawing shapes and text on the screen.
"""

# Authors: Dan McCloy <drmccloy@uw.edu>
#          Eric Larson <larsoner@uw.edu>
#          Ross Maddox <rkmaddox@uw.edu>
#
# License: BSD (3-clause)

import re
import warnings
from contextlib import contextmanager
from ctypes import POINTER, c_char, c_float, c_int, cast, create_string_buffer, pointer

import numpy as np

from .._utils import check_units, logger

# TODO: Should eventually follow
# https://pyglet.readthedocs.io/en/latest/programming_guide/rendering.html
# Should have cleaner programs, less manual GL usage, etc.
# For a start, see this (but requires Pyglet 2+):
# https://gist.github.com/larsoner/1a9bbdc26bc3a3af1de8e5033fb99891
# Even better, we could refactor to use native Pyglet shapes:
# https://github.com/pyglet/pyglet/blob/master/examples/shapes.py


def _convert_color(color, byte=True):
    """Convert 3- or 4-element color into OpenGL usable color."""
    from matplotlib.colors import colorConverter

    color = (0.0, 0.0, 0.0, 0.0) if color is None else color
    color = 255 * np.array(colorConverter.to_rgba(color))
    color = color.astype(np.uint8)
    if not byte:
        color = tuple((color / 255.0).astype(np.float32))
    else:
        color = tuple(int(c) for c in color)
    return color


@contextmanager
def _element_array_buffer(buffer_id):
    from pyglet import gl

    old = (gl.GLint * 1)()
    gl.glGetIntegerv(gl.GL_ELEMENT_ARRAY_BUFFER_BINDING, old)
    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, buffer_id)
    try:
        yield
    finally:
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, list(old)[0])


@contextmanager
def _array_buffer(buffer_id):
    from pyglet import gl

    old = (gl.GLint * 1)()
    gl.glGetIntegerv(gl.GL_ARRAY_BUFFER_BINDING, old)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer_id)
    try:
        yield
    finally:
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, list(old)[0])


@contextmanager
def _use_program(program):
    from pyglet import gl

    old = (gl.GLint * 1)()
    gl.glGetIntegerv(gl.GL_CURRENT_PROGRAM, old)
    gl.glUseProgram(program)
    try:
        yield
    finally:
        gl.glUseProgram(list(old)[0])


##############################################################################
# Text


class Text:
    """A text object.

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
    attr : bool
        Should the text be interpreted with pyglet's ``decode_attributed``
        method? This allows inline formatting for text color, e.g.,
        ``'This is {color (255, 0, 0, 255)}red text'``. If ``attr=True``, the
        values of ``font_name``, ``font_size``, and ``color`` are automatically
        prepended to ``text`` (though they will be overridden by any inline
        formatting within ``text`` itself).

    Returns
    -------
    text : instance of Text
        The text object.
    """

    def __init__(
        self,
        ec,
        text,
        pos=(0, 0),
        color="white",
        font_name="Arial",
        font_size=24,
        height=None,
        width="auto",
        anchor_x="center",
        anchor_y="center",
        units="norm",
        wrap=False,
        attr=True,
    ):
        import pyglet

        pyglet_2 = hasattr(pyglet, "__version__")
        pos = np.array(pos)[:, np.newaxis]
        pos = ec._convert_units(pos, units, "pix")[:, 0]
        if width == "auto":
            width = float(ec.window_size_pix[0]) * 0.8
        elif isinstance(width, str):
            raise ValueError('"width", if str, must be "auto"')
        self._attr = attr
        kwargs = dict(width=width, height=height, multiline=wrap, dpi=int(ec.dpi))
        if pyglet_2:
            # Really needed to work around
            # https://github.com/pyglet/pyglet/issues/1359
            kwargs.update(anchor_x=anchor_x, anchor_y=anchor_y, x=pos[0], y=pos[1])
        elif wrap:
            text = text + "\n "  # weird Pyglet bug
        if self._attr:
            preamble = (
                f"{{font_name '{font_name}'}}"
                f"{{font_size {font_size}}}"
                f"{{color {_convert_color(color)}}}"
            )
            doc = pyglet.text.decode_attributed(preamble + text)
            self._text = pyglet.text.layout.TextLayout(doc, **kwargs)
        else:
            self._text = pyglet.text.Label(text, **kwargs)
            self._text.color = _convert_color(color)
            self._text.font_name = font_name
            self._text.font_size = font_size
        if not pyglet_2:
            self._text.x = pos[0]
            self._text.y = pos[1]
            self._text.anchor_x = anchor_x
            self._text.anchor_y = anchor_y

    def set_color(self, color):
        """Set the text color

        Parameters
        ----------
        color : matplotlib Color | None
            The color. Use None for no color.
        """
        if self._attr:
            self._text.document.set_style(
                0, len(self._text.document.text), {"color": _convert_color(color)}
            )
        else:
            self._text.color = _convert_color(color)

    def draw(self):
        """Draw the object to the display buffer"""
        self._text.draw()


##############################################################################
# Triangulations

tri_vert = """\
#version 150 core

in vec2 a_position;
uniform mat4 u_view;

void main()
{
    gl_Position = u_view * vec4(a_position, 0.0, 1.0);
}
"""

tri_frag = """
#version 150 core

uniform vec4 u_color;
out vec4 FragColor;

void main()
{
    FragColor = u_color;
}
"""


def _check_log(obj, func):
    log = create_string_buffer(4096)
    ptr = cast(pointer(log), POINTER(c_char))
    func(obj, 4096, pointer(c_int()), ptr)
    message = log.value
    message = message.decode()
    if (
        message.startswith("No errors")
        or re.match(".*shader was successfully compiled.*", message)
        or message == "Vertex shader(s) linked, fragment shader(s) linked.\n"
    ):
        pass
    elif message:
        raise RuntimeError(message)


def _create_program(ec, vert, frag):
    from pyglet import gl

    program = gl.glCreateProgram()

    vertex = gl.glCreateShader(gl.GL_VERTEX_SHADER)
    buf = create_string_buffer(vert.encode("ASCII"))
    ptr = cast(pointer(pointer(buf)), POINTER(POINTER(c_char)))
    gl.glShaderSource(vertex, 1, ptr, None)
    gl.glCompileShader(vertex)
    _check_log(vertex, gl.glGetShaderInfoLog)

    fragment = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
    buf = create_string_buffer(frag.encode("ASCII"))
    ptr = cast(pointer(pointer(buf)), POINTER(POINTER(c_char)))
    gl.glShaderSource(fragment, 1, ptr, None)
    gl.glCompileShader(fragment)
    _check_log(fragment, gl.glGetShaderInfoLog)

    gl.glAttachShader(program, vertex)
    gl.glAttachShader(program, fragment)
    gl.glLinkProgram(program)
    _check_log(program, gl.glGetProgramInfoLog)

    gl.glDetachShader(program, vertex)
    gl.glDetachShader(program, fragment)

    # Set the view matrix
    with _use_program(program):
        loc = gl.glGetUniformLocation(program, b"u_view")
        view = ec.window_size_pix
        view = np.diag([2.0 / view[0], 2.0 / view[1], 1.0, 1.0])
        view[-1, :2] = -1
        view = view.astype(np.float32).ravel()
        gl.glUniformMatrix4fv(loc, 1, False, (c_float * 16)(*view))
    return program


class _Triangular:
    """Super class for objects that use triangulations and/or lines"""

    def __init__(self, ec, fill_color, line_color, line_width, line_loop):
        from pyglet import gl

        self._ec = ec
        self._line_width = line_width
        self._line_loop = line_loop  # whether or not lines drawn are looped

        # initialize program and shaders
        self._program = _create_program(ec, tri_vert, tri_frag)

        self._counts = dict()
        self._colors = dict()
        self._buffers = dict()
        self._points = dict()
        self._tris = dict()
        with _use_program(self._program):
            for kind in ("line", "fill"):
                self._counts[kind] = 0
                self._colors[kind] = (0.0, 0.0, 0.0, 0.0)
                self._buffers[kind] = dict(array=gl.GLuint())
                gl.glGenBuffers(1, pointer(self._buffers[kind]["array"]))
                if kind == "fill":
                    self._buffers[kind]["index"] = gl.GLuint()
                    gl.glGenBuffers(1, pointer(self._buffers[kind]["index"]))

        self.set_fill_color(fill_color)
        self.set_line_color(line_color)
        self._vao = gl.GLuint()
        gl.glGenVertexArrays(1, pointer(self._vao))

    def _set_points(self, points, kind, tris):
        """Set fill and line points."""
        from pyglet import gl

        if points is None:
            self._counts[kind] = 0
        points = np.asarray(points, dtype=np.float32, order="C")
        assert points.ndim == 2 and points.shape[1] == 2
        array_count = points.size // 2 if kind == "line" else points.size
        if kind == "fill":
            assert tris is not None
            tris = np.asarray(tris, dtype=np.uint32, order="C")
            assert tris.ndim == 1 and tris.size % 3 == 0
            tris.shape = (-1, 3)
            assert (tris < len(points)).all()
            self._tris[kind] = tris
            del tris
        self._points[kind] = points
        del points
        with _use_program(self._program), _array_buffer(self._buffers[kind]["array"]):
            gl.glBufferData(
                gl.GL_ARRAY_BUFFER,
                self._points[kind].size * 4,
                self._points[kind].tobytes(),
                gl.GL_DYNAMIC_DRAW,
            )
            if kind == "line":
                self._counts[kind] = array_count
            if kind == "fill":
                self._counts[kind] = self._tris[kind].size
                with _element_array_buffer(self._buffers[kind]["index"]):
                    gl.glBufferData(
                        gl.GL_ELEMENT_ARRAY_BUFFER,
                        self._tris[kind].size * 4,
                        self._tris[kind].tobytes(),
                        gl.GL_STATIC_DRAW,
                    )

    def _set_fill_points(self, points, tris):
        self._set_points(points, "fill", tris)

    def _set_line_points(self, points):
        self._set_points(points, "line", None)

    def set_fill_color(self, fill_color):
        """Set the object color

        Parameters
        ----------
        fill_color : matplotlib Color | None
            The fill color. Use None for no fill.
        """
        self._colors["fill"] = _convert_color(fill_color, byte=False)

    def set_line_color(self, line_color):
        """Set the object color

        Parameters
        ----------
        line_color : matplotlib Color | None
            The fill color. Use None for no fill.
        """
        self._colors["line"] = _convert_color(line_color, byte=False)

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
            raise ValueError("line_width must be between 0 and 10")
        self._line_width = line_width

    def draw(self):
        """Draw the object to the display buffer."""
        from pyglet import gl

        gl.glBindVertexArray(self._vao)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        for kind in ("fill", "line"):
            if self._counts[kind] == 0:
                print(f"Skipping {kind} for {self}")
                continue
            with (
                _use_program(self._program),
                _array_buffer(self._buffers[kind]["array"]),
            ):
                loc_pos = gl.glGetAttribLocation(self._program, b"a_position")
                gl.glEnableVertexAttribArray(loc_pos)
                gl.glVertexAttribPointer(loc_pos, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, 0)
                loc_col = gl.glGetUniformLocation(self._program, b"u_color")
                gl.glUniform4f(loc_col, *self._colors[kind])
                if kind == "line":
                    if self._line_width <= 0.0:
                        continue
                    try:
                        gl.glLineWidth(self._line_width)
                    except gl.lib.GLException:  # on macOS for example only 1 works
                        gl.glLineWidth(1.0)
                    if self._line_loop:
                        mode = gl.GL_LINE_LOOP
                    else:
                        mode = gl.GL_LINE_STRIP
                    gl.glDrawArrays(mode, 0, self._counts[kind])
                else:
                    with _element_array_buffer(self._buffers[kind]["index"]):
                        gl.glDrawElements(
                            gl.GL_TRIANGLES,
                            self._counts[kind],
                            gl.GL_UNSIGNED_INT,
                            0,
                        )
                gl.glDisableVertexAttribArray(loc_pos)
        gl.glBindVertexArray(0)


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

    def __init__(
        self,
        ec,
        coords,
        units="norm",
        line_color="white",
        line_width=1.0,
        line_loop=False,
    ):
        super().__init__(
            ec,
            fill_color=None,
            line_color=line_color,
            line_width=line_width,
            line_loop=line_loop,
        )
        self.set_coords(coords, units)
        self.set_line_color(line_color)

    def set_coords(self, coords, units="norm"):
        """Set line coordinates

        Parameters
        ----------
        coords : array-like
            2 x N set of X, Y coordinates.
        units : str
            Units to use.
        """
        check_units(units)
        coords = np.array(coords, dtype=float)
        if coords.ndim == 1:
            coords = coords[:, np.newaxis]
        if coords.ndim != 2 or coords.shape[0] != 2:
            raise ValueError(
                "coords must be a vector of length 2, or an "
                "array with 2 dimensions (with first dimension "
                "having length 2"
            )
        self._set_line_points(self._ec._convert_units(coords, units, "pix").T)


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

    def __init__(
        self,
        ec,
        coords,
        units="norm",
        fill_color="white",
        line_color=None,
        line_width=1.0,
    ):
        super().__init__(
            ec,
            fill_color=fill_color,
            line_color=line_color,
            line_width=line_width,
            line_loop=True,
        )
        self.set_coords(coords, units)
        self.set_fill_color(fill_color)

    def set_coords(self, coords, units="norm"):
        """Set triangle coordinates

        Parameters
        ----------
        coords : array-like
            2 x 3 set of X, Y coordinates.
        units : str
            Units to use.
        """
        check_units(units)
        coords = np.array(coords, dtype=float)
        if coords.shape != (2, 3):
            raise ValueError(
                "coords must be an array of shape (2, 3), got %s" % (coords.shape,)
            )
        points = self._ec._convert_units(coords, units, "pix")
        points = points.T
        self._set_fill_points(points, [0, 1, 2])
        self._set_line_points(points)


class Rectangle(_Triangular):
    """A rectangle.

    Parameters
    ----------
    ec : instance of ExperimentController
        Parent EC.
    pos : array-like
        4-element array-like with X, Y center and width, height where x and y
        are coordinates of the center.
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

    def __init__(
        self, ec, pos, units="norm", fill_color="white", line_color=None, line_width=1.0
    ):
        super().__init__(
            ec,
            fill_color=fill_color,
            line_color=line_color,
            line_width=line_width,
            line_loop=True,
        )
        self.set_pos(pos, units)

    def set_pos(self, pos, units="norm"):
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
            raise ValueError("pos must be a 4-element array-like vector")
        self._pos = pos
        w = self._pos[2]
        h = self._pos[3]
        points = np.array(
            [
                [-w / 2.0, -h / 2.0],
                [-w / 2.0, h / 2.0],
                [w / 2.0, h / 2.0],
                [w / 2.0, -h / 2.0],
            ]
        ).T
        points += np.array(self._pos[:2])[:, np.newaxis]
        points = self._ec._convert_units(points, units, "pix")
        points = points.T
        self._set_fill_points(points, [0, 1, 2, 0, 2, 3])
        self._set_line_points(points)  # all 4 points used for line drawing


class Diamond(_Triangular):
    """A diamond.

    Parameters
    ----------
    ec : instance of ExperimentController
        Parent EC.
    pos : array-like
        4-element array-like with X, Y center and width, height where x and y
        are coordinates of the center.
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

    def __init__(
        self, ec, pos, units="norm", fill_color="white", line_color=None, line_width=1.0
    ):
        super().__init__(
            ec,
            fill_color=fill_color,
            line_color=line_color,
            line_width=line_width,
            line_loop=True,
        )
        self.set_pos(pos, units)

    def set_pos(self, pos, units="norm"):
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
            raise ValueError("pos must be a 4-element array-like vector")
        self._pos = pos
        w = self._pos[2]
        h = self._pos[3]
        points = np.array(
            [[w / 2.0, 0.0], [0.0, h / 2.0], [-w / 2.0, 0.0], [0.0, -h / 2.0]]
        ).T
        points += np.array(self._pos[:2])[:, np.newaxis]
        points = self._ec._convert_units(points, units, "pix")
        points = points.T
        self._set_fill_points(points, [0, 1, 2, 0, 2, 3])
        self._set_line_points(points)


class Circle(_Triangular):
    """A circle or ellipse.

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

    def __init__(
        self,
        ec,
        radius=1,
        pos=(0, 0),
        units="norm",
        n_edges=200,
        fill_color="white",
        line_color=None,
        line_width=1.0,
    ):
        super().__init__(
            ec,
            fill_color=fill_color,
            line_color=line_color,
            line_width=line_width,
            line_loop=True,
        )
        if not isinstance(n_edges, int):
            raise TypeError("n_edges must be an int")
        if n_edges < 4:
            raise ValueError("n_edges must be >= 4 for a reasonable circle")
        self._n_edges = n_edges

        # construct triangulation (never changes so long as n_edges is fixed)
        tris = [[0, ii + 1, ii + 2] for ii in range(n_edges)]
        tris = np.concatenate(tris)
        tris[-1] = 1  # fix wrap for last triangle
        self._orig_tris = tris

        # need to set a dummy value here so recalculation doesn't fail
        self._radius = np.array([1.0, 1.0])
        self.set_pos(pos, units)
        self.set_radius(radius, units)

    def set_radius(self, radius, units="norm"):
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
            raise ValueError("radius must be a 1- or 2-element array-like vector")
        if radius.size == 1:
            radius = np.r_[radius, radius]
        # convert to pixel (OpenGL) units
        self._radius = self._ec._convert_units(radius[:, np.newaxis], units, "pix")[
            :, 0
        ]
        # need to subtract center position
        ctr = self._ec._convert_units(np.zeros((2, 1)), units, "pix")[:, 0]
        self._radius -= ctr
        self._recalculate()

    def set_pos(self, pos, units="norm"):
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
            raise ValueError("pos must be a 2-element array-like vector")
        # convert to pixel (OpenGL) units
        self._pos = self._ec._convert_units(pos[:, np.newaxis], units, "pix")[:, 0]
        self._recalculate()

    def _recalculate(self):
        """Helper to recalculate point coordinates"""
        edges = self._n_edges
        arg = 2 * np.pi * (np.arange(edges) / float(edges))
        points = np.array(
            [self._radius[0] * np.cos(arg), self._radius[1] * np.sin(arg)]
        )
        points = np.c_[np.zeros((2, 1)), points]  # prepend the center
        points += np.array(self._pos[:2], dtype=float)[:, np.newaxis]
        points = points.T
        self._set_fill_points(points, self._orig_tris)
        self._set_line_points(points[1:])  # omit center point for lines


class ConcentricCircles:
    """A set of filled concentric circles drawn without edges.

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

    def __init__(
        self, ec, radii=(0.2, 0.05), pos=(0, 0), units="norm", colors=("w", "k")
    ):
        radii = np.array(radii, float)
        if radii.ndim != 1:
            raise ValueError("radii must be 1D")
        if not isinstance(colors, (tuple, list)):
            raise TypeError("colors must be a tuple, list, or array")
        if len(colors) != len(radii):
            raise ValueError("colors and radii must be the same length")
        # need to set a dummy value here so recalculation doesn't fail
        self._circles = [
            Circle(ec, r, pos, units, fill_color=c, line_width=0)
            for r, c in zip(radii, colors)
        ]

    def __len__(self):
        return len(self._circles)

    def set_pos(self, pos, units="norm"):
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

    def set_radius(self, radius, idx, units="norm"):
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

    def set_radii(self, radii, units="norm"):
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
            raise ValueError(f"radii must contain exactly {len(self)} radii")
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
        """
        self._circles[idx].set_fill_color(color)

    def set_colors(self, colors):
        """Set the color of each circle.

        Parameters
        ----------
        colors : list or tuple of matplotlib Colors
            Must be of type list or tuple, and contain the same number of
            colors as the number of circles.
        """
        if not isinstance(colors, (tuple, list)) or len(colors) != len(self):
            raise ValueError(f"colors must be a list or tuple with {len(self)} colors")
        for idx, color in enumerate(colors):
            self.set_color(color, idx)

    def draw(self):
        """Draw the fixation dot."""
        for circle in self._circles:
            circle.draw()


class FixationDot(ConcentricCircles):
    """A reasonable centered fixation dot.

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

    def __init__(self, ec, colors=("w", "k")):
        if len(colors) != 2:
            raise ValueError("colors must have length 2")
        super().__init__(ec, radii=[0.2, 0.2], pos=[0, 0], units="deg", colors=colors)
        self.set_radius(1, 1, units="pix")


class ProgressBar:
    """A progress bar that can be displayed between sections.

    This uses two rectangles, one outline, and one solid to show how much
    progress has been made in the experiment.

    Parameters
    ----------
    ec : instance of ExperimentController
        Parent EC.
    pos : array-like
        4-element array-like with X, Y center and width, height where x and y
        are coordinates of the box center.
    units : str
        Units to use. These will apply to all spatial aspects of the drawing.
        Must be either ``'norm'`` or ``'pix'``.
    colors : list or tuple of matplotlib Colors
        Colors to fill and outline the bar respectively. Defaults to green and
        white.
    """

    def __init__(self, ec, pos, units="norm", colors=("g", "w")):
        self._ec = ec
        if len(colors) != 2:
            raise ValueError("colors must have length 2")
        if units not in ["norm", "pix"]:
            raise ValueError("units must be either 'norm' or 'pix'")

        pos = np.array(pos, dtype=float)
        self._pos = pos
        self._width = pos[2]
        self._units = units

        # initialize the bar with zero progress
        self._pos_bar = pos.copy()
        self._pos_bar[0] -= self._width * 0.5
        self._init_x = self._pos_bar[0]
        self._pos_bar[2] = 0

        self._rectangles = [
            Rectangle(ec, self._pos_bar, units, colors[0], None),
            Rectangle(ec, self._pos, units, None, colors[1]),
        ]

    def update_bar(self, percent):
        """Update the progress of the bar.

        Parameters
        ----------
        percent: float
            The percentage of the bar to be filled. Must be between 0 and 1.
        """
        if percent > 100 or percent < 0:
            raise ValueError("percent must be a float between 0 and 100")
        self._pos_bar[2] = percent * self._width / 100.0
        self._pos_bar[0] = self._init_x + self._pos_bar[2] * 0.5
        self._rectangles[0].set_pos(self._pos_bar, self._units)

    def draw(self):
        """Draw the progress bar."""
        for rectangle in self._rectangles:
            rectangle.draw()


##############################################################################
# Image display


class RawImage:
    """Create image from array for on-screen display.

    Parameters
    ----------
    ec : instance of ExperimentController
        Parent EC.
    image_buffer : array
        Array, shape (N, M[, 3/4]). Color values should range between 0 and 1.
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

    def __init__(self, ec, image_buffer, pos=(0, 0), scale=1.0, units="norm"):
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
            raise TypeError("image_buffer must be np.float64 or np.uint8")
        if image_buffer.dtype == np.float64:
            if image_buffer.max() > 1 or image_buffer.min() < 0:
                raise ValueError("all float values must be between 0 and 1")
            image_buffer = (image_buffer * 255).astype("uint8")
        assert image_buffer.dtype == np.uint8
        if image_buffer.ndim == 2:  # grayscale
            image_buffer = np.tile(image_buffer[..., np.newaxis], (1, 1, 3))
        dims = image_buffer.shape
        if len(dims) != 3 or dims[2] not in [3, 4]:
            raise ValueError(f"image_buffer incorrect shape: {image_buffer.shape}")
        fmt = "RGB" if dims[2] == 3 else "RGBA"
        self._sprite = sprite.Sprite(
            image.ImageData(
                dims[1], dims[0], fmt, image_buffer.tobytes(), -dims[1] * dims[2]
            )
        )

    def set_pos(self, pos, units="norm"):
        """Set image position.

        Parameters
        ----------
        pos : array-like
            2-element array-like with X, Y (center) arguments.
        units : str
            Units to use. See ``check_units`` for options.
        """
        pos = np.array(pos, float)
        if pos.ndim != 1 or pos.size != 2:
            raise ValueError("pos must be a 2-element array")
        pos = np.reshape(pos, (2, 1))
        self._pos = self._ec._convert_units(pos, units, "pix").ravel()

    @property
    def bounds(self):
        """Left, Right, Bottom, Top (in pixels) of the image."""
        pos = np.array(self._pos, float)
        size = np.array([self._sprite.width, self._sprite.height], float)
        bounds = np.concatenate((pos - size / 2.0, pos + size / 2.0))
        return bounds[[0, 2, 1, 3]]

    @property
    def scale(self):
        return self._scale

    def set_scale(self, scale):
        """Create image from array for on-screen display.

        Parameters
        ----------
        scale : float
            The scale factor. 1 is native size (pixel-to-pixel), 2 is twice as
            large, etc.
        """
        scale = float(scale)
        self._scale = scale
        self._sprite.scale = self._scale

    def draw(self):
        """Draw the image to the buffer"""
        self._sprite.scale = self._scale
        pos = self._pos - [self._sprite.width / 2.0, self._sprite.height / 2.0]
        if hasattr(self._sprite, "z"):  # Pyglet 2+
            self._sprite.position = (*pos, self._sprite.z)
        else:
            self._sprite.position = pos
        self._sprite.draw()

    def get_rect(self, units="norm"):
        """X, Y center, Width, Height of image.

        Parameters
        ----------
        units : str
            Units to use for the position. See ``check_units`` for options.

        Returns
        -------
        rect : ndarray
            The rect.
        """
        # left,right,bottom,top
        lrbt = self._ec._convert_units(self.bounds.reshape(2, -1), fro="pix", to=units)
        center = self._ec._convert_units(self._pos.reshape(2, -1), fro="pix", to=units)
        width_height = np.diff(lrbt, axis=-1)
        return np.squeeze(np.concatenate([center, width_height]))


tex_vert = """\
#version 150 core

in vec2 a_position;
in vec2 a_texcoord;
uniform mat4 u_view;
out vec2 v_texcoord;

void main()
{
    gl_Position = u_view * vec4(a_position, 0.0, 1.0);
    v_texcoord = a_texcoord;
}
"""

tex_frag = """\
#version 150 core

uniform sampler2D u_texture;
in vec2 v_texcoord;
out vec4 FragColor;

void main()
{
    FragColor = texture(u_texture, v_texcoord);
    FragColor.a = 1.0;
}
"""


class Video:
    """Read video file and draw it to the screen.

    Parameters
    ----------
    ec : instance of expyfun.ExperimentController
    file_name : str
        the video file path
    pos : array-like
        2-element array-like with X, Y elements.
    units : str
        Units to use for the position. See ``check_units`` for options.
    scale : float | str
        The scale factor. 1 is native size (pixel-to-pixel), 2 is twice as
        large, etc. If scale is a string, it must be either ``'fill'``
        (which ensures the entire ``ExperimentController`` window is
        covered by the video, at the expense of some parts of the video
        potentially being offscreen), or ``'fit'`` (which scales maximally
        while ensuring none of the video is offscreen, and may result in
        letterboxing or pillarboxing).
    center : bool
        If ``False``, the elements of ``pos`` specify the position of the lower
        left corner of the video frame; otherwise they position the center of
        the frame.
    visible : bool
        Whether to show the video when initialized. Can be toggled later using
        `Video.set_visible` method.

    Notes
    -----
    This is a somewhat pared-down implementation of video playback. Looping is
    not available, and the audio stream from the video file is discarded.
    Timing of individual frames is relegated to the pyglet media player's
    internal clock. Recommended for use only in paradigms where the relative
    timing of audio and video are unimportant (e.g., if the video is merely
    entertainment for the participant during a passive auditory task).
    """

    def __init__(
        self,
        ec,
        file_name,
        pos=(0, 0),
        units="norm",
        scale=1.0,
        center=True,
        visible=True,
    ):
        from pyglet import gl
        from pyglet.media import Player, load

        self._ec = ec
        # On Windows, the default is unaccelerated WMF, which is terribly slow.
        decoder = None
        try:
            from pyglet.media.codecs.ffmpeg import FFmpegDecoder

            decoder = FFmpegDecoder()
        except Exception as exc:
            warnings.warn(
                "FFmpeg decoder could not be instantiated, decoding "
                f"performance could be compromised:\n{exc}"
            )
        self._source = load(file_name, decoder=decoder)
        self._player = Player()
        with warnings.catch_warnings(record=True):  # deprecated eos_action
            self._player.queue(self._source)
        self._player._audio_player = None
        frame_rate = self.frame_rate
        if frame_rate is None:
            logger.warning("Frame rate could not be determined")
            frame_rate = 60.0
        self._dt = 1.0 / frame_rate
        self._playing = False
        self._finished = False
        self._pos = pos
        self._units = units
        self._center = center
        self.set_scale(scale)  # also calls set_pos
        self._visible = visible

        self._program = _create_program(ec, tex_vert, tex_frag)
        gl.glUseProgram(self._program)
        self._buffers = dict()
        for key in ("position", "texcoord"):
            self._buffers[key] = gl.GLuint(0)
            gl.glGenBuffers(1, pointer(self._buffers[key]))
        a, b, c, d = [(0, 1), (1, 1), (1, 0), (0, 0)]
        tex = np.array([a, b, c, a, c, d], np.float32)
        with _array_buffer(self._buffers["texcoord"]):
            gl.glBufferData(
                gl.GL_ARRAY_BUFFER, tex.nbytes, tex.tobytes(), gl.GL_DYNAMIC_DRAW
            )
        gl.glUseProgram(0)
        self._vao = gl.GLuint()
        gl.glGenVertexArrays(1, pointer(self._vao))

    def play(self, auto_draw=True, audio=False):
        """Play video from current position.

        Parameters
        ----------
        auto_draw : bool
            If True, add ``self.draw`` to ``ec.on_every_flip``.
        audio : bool
            Whether to play the audio stream. Only works if the
            :class:`~expyfun.ExperimentController` was instantiated with
            ``audio_controller=dict(TYPE="sound_card", SOUND_CARD_BACKEND="pyglet")``,
            and will raise an error if that is not the case.

        Returns
        -------
        time : float
            The timestamp (on the parent ``ExperimentController`` timeline) at
            which ``play()`` was called.
        """
        if audio:
            if self._ec.audio_type != "pyglet":
                raise ValueError(
                    "Cannot play a video's audio stream unless the audio type is "
                    "'sound_card' and the backend is 'pyglet'."
                )
            self._player.volume = 1.0
        else:
            self._player.volume = 0.0
        if not self._playing:
            if auto_draw:
                self._ec.call_on_every_flip(self.draw)
            self._player.play()
            self._playing = True
        else:
            warnings.warn(
                "ExperimentController.video.play() called when already playing."
            )
        return self._ec.get_time()

    def pause(self):
        """Halt video playback.

        Returns
        -------
        time : float
            The timestamp (on the parent ``ExperimentController`` timeline) at
            which ``pause()`` was called.
        """
        if self._playing:
            try:
                idx = self._ec.on_every_flip_functions.index(self.draw)
            except ValueError:  # not auto_draw
                pass
            else:
                self._ec.on_every_flip_functions.pop(idx)
            self._player.pause()
            self._playing = False
        else:
            warnings.warn(
                "ExperimentController.video.pause() called when already paused."
            )
        return self._ec.get_time()

    def _delete(self):
        """Halt video playback and remove player."""
        if self._playing:
            self.pause()
        self._player.delete()

    def set_scale(self, scale=1.0):
        """Set video scale.

        Parameters
        ----------
        scale : float | str
            The scale factor. 1 is native size (pixel-to-pixel), 2 is twice as
            large, etc. If scale is a string, it must be either ``'fill'``
            (which ensures the entire ``ExperimentController`` window is
            covered by the video, at the expense of some parts of the video
            potentially being offscreen), or ``'fit'`` (which scales maximally
            while ensuring none of the video is offscreen, which may result in
            letterboxing).
        """
        if isinstance(scale, str):
            _scale = self._ec.window_size_pix / np.array(
                (self.source_width, self.source_height), dtype=float
            )
            if scale == "fit":
                scale = _scale.min()
            elif scale == "fill":
                scale = _scale.max()
        self._scale = float(scale)  # allows [1, 1., '1']; others: ValueError
        if self._scale <= 0:
            raise ValueError("Video scale factor must be strictly positive.")
        self.set_pos(self._pos, self._units, self._center)

    def set_pos(self, pos, units="norm", center=True):
        """Set video position.

        Parameters
        ----------
        pos : array-like
            2-element array-like with X, Y elements.
        units : str
            Units to use for the position. See ``check_units`` for options.
        center : bool
            If ``False``, the elements of ``pos`` specify the position of the
            lower left corner of the video frame; otherwise they position the
            center of the frame.
        """
        pos = np.array(pos, float)
        if pos.size != 2:
            raise ValueError("pos must be a 2-element array")
        pos = np.reshape(pos, (2, 1))
        pix = self._ec._convert_units(pos, units, "pix").ravel()
        offset = np.array((self.width, self.height)) // 2 if center else 0
        self._pos = pos
        self._actual_pos = pix - offset
        self._pos_unit = units
        self._pos_centered = center

    def _draw(self):
        from pyglet import gl

        tex = self._player.texture
        gl.glUseProgram(self._program)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(tex.target, tex.id)
        gl.glBindVertexArray(self._vao)
        x, y = self._actual_pos
        w = self.source_width * self._scale
        h = self.source_height * self._scale
        a, b, c, d = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        pos = np.array([a, b, c, a, c, d], np.float32)
        with _array_buffer(self._buffers["position"]):
            gl.glBufferData(
                gl.GL_ARRAY_BUFFER, pos.nbytes, pos.tobytes(), gl.GL_DYNAMIC_DRAW
            )
            loc_pos = gl.glGetAttribLocation(self._program, b"a_position")
            gl.glEnableVertexAttribArray(loc_pos)
            gl.glVertexAttribPointer(loc_pos, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, 0)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._buffers["texcoord"])
            loc_tex = gl.glGetAttribLocation(self._program, b"a_texcoord")
            gl.glEnableVertexAttribArray(loc_tex)
            gl.glVertexAttribPointer(loc_tex, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, 0)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
            gl.glDisableVertexAttribArray(loc_pos)
            gl.glDisableVertexAttribArray(loc_tex)
        gl.glBindTexture(tex.target, 0)
        gl.glBindVertexArray(0)
        gl.glUseProgram(0)

    def draw(self):
        """Draw the video texture to the screen buffer."""
        done = False
        if self._player.source is None:
            done = True
        if not done:
            self._player.update_texture()
            # detect end-of-stream to prevent pyglet from hanging:
            if self._eos:
                done = True
            elif self._visible:
                self._draw()
        if done:
            self._finished = True
            self.pause()
        self._ec.check_force_quit()

    def set_visible(self, show, flip=False):
        """Show/hide the video frame.

        Parameters
        ----------
        show : bool
            Show or hide.
        flip : bool
            If True, flip after showing or hiding.
        """
        if show:
            self._visible = True
            self._draw()
        else:
            self._visible = False
            self._ec.flip()
        if flip:
            self._ec.flip()

    # PROPERTIES
    @property
    def _eos(self):
        done = self._player.source is None
        ts = self._source.get_next_video_timestamp()
        dur = self._source._duration
        return done or ts is None or ts >= dur

    @property
    def playing(self):
        return self._playing

    @property
    def finished(self):
        return self._finished

    @property
    def position(self):
        return np.squeeze(self._pos)

    @property
    def scale(self):
        return self._scale

    @property
    def duration(self):
        return self._source.duration

    @property
    def frame_rate(self):
        return self._source.video_format.frame_rate

    @property
    def dt(self):
        return self._dt

    @property
    def time(self):
        return self._player.time

    @property
    def width(self):
        return self.source_width * self._scale

    @property
    def height(self):
        return self.source_height * self._scale

    @property
    def source_width(self):
        return self._source.video_format.width

    @property
    def source_height(self):
        return self._source.video_format.height

    @property
    def time_offset(self):
        return self._ec.get_time() - self._player.time
