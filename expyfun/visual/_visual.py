"""Tools for drawing shapes and text on the screen"""

# Authors: Dan McCloy <drmccloy@uw.edu>
#          Eric Larson <larsoner@uw.edu>
#
# License: BSD (3-clause)

import numpy as np
import pyglet
from matplotlib.colors import colorConverter


def _convert_color(color, n_pts):
    """Convert 3- or 4-element color into OpenGL usable color"""
    color = np.tile(255 * np.array(colorConverter.to_rgba(color)), n_pts)
    color = color.astype(np.uint8)
    return color




class Rectangle(object):
    def __init__(self, ec, width, height, x=0., y=0., fill_color='white'):
        # do this in normalized units, then convert
        points = np.array([[-width / 2., -height / 2.],
                           [-width / 2., height / 2.],
                           [width / 2., height / 2.],
                           [width / 2., -height / 2.]]).T
        points += np.array([x, y])[:, np.newaxis]
        points = ec._convert_units(points, 'norm', 'pix')
        self.points = points.ravel().astype(int)
        self.color = _convert_color(fill_color, 4)

    def draw(self):
        pyglet.graphics.draw_indexed(4, pyglet.gl.GL_TRIANGLES,
                                     [0, 1, 2, 0, 2, 3],
                                     ('v2i', self.points),
                                     ('c4B', self.color))


class Text(object):
    def __init__(self, ec, text):
        self._text = pyglet.text.HTMLLabel(text, width=float(ec.size_pix[0]),
                                           multiline=True, dpi=int(ec.dpi),
                                           anchor_x='center',
                                           anchor_y='center')
        self.draw = self._text.draw
