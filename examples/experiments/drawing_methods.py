"""
==========================
Experiment drawing methods
==========================

expyfun provides multiple methods for drawing simple screen objects.
"""
# Author: Eric Larson <larsoner@uw.edu>
#
# License: BSD (3-clause)

import numpy as np

import expyfun.analyze as ea
from expyfun import ExperimentController, visual

print(__doc__)


with ExperimentController(
    "test",
    session="1",
    participant="2",
    full_screen=False,
    window_size=[600, 600],
    output_dir=None,
    version="dev",
) as ec:
    ec.screen_text("hello")

    # make an image with  alpha the x-dimension (columns), RGB upward
    img_buffer = np.zeros((120, 100, 4))
    img_buffer[:, :50, 3] = 1.0
    img_buffer[:, 50:, 3] = 0.5
    img_buffer[0] = 1
    for ii in range(3):
        img_buffer[ii * 40 : (ii + 1) * 40, :, ii] = 1.0
    img = visual.RawImage(ec, img_buffer, scale=2.0)

    # make a line, rectangle, diamond, and circle
    line = visual.Line(
        ec,
        [[-2, 2, 2, -2], [-2, 2, -2, -2]],
        units="deg",
        line_color="w",
        line_width=2.0,
    )
    rect = visual.Rectangle(ec, [0, 0, 2, 2], units="deg", fill_color="y")
    diamond = visual.Diamond(
        ec,
        [0, 0, 4, 4],
        units="deg",
        fill_color=None,
        line_color="gray",
        line_width=2.0,
    )
    circle = visual.Circle(
        ec, 1, units="deg", line_color="w", fill_color="k", line_width=2.0
    )

    # do the drawing, then flip
    for obj in [img, line, rect, diamond, circle]:
        obj.draw()

    screenshot = ec.screenshot()  # must be called *before* the flip
    ec.flip()
    ec.wait_for_presses(0.5)

ea.plot_screen(screenshot)
