"""
==========================
Experiment drawing methods
==========================

expyfun provides multiple methods for drawing simple screen objects.
"""
# Author: Eric Larson <larsoner@uw.edu>
#
# License: BSD (3-clause)

print(__doc__)

import numpy as np
from expyfun import visual, ExperimentController
import expyfun.analyze as ea


with ExperimentController('test', session='1', participant='2',
                          full_screen=False, window_size=[600, 600],
                          output_dir=None) as ec:
    ec.screen_text('hello')

    # make an image with  alpha the x-dimension (columns), RGB upward
    img_buffer = np.zeros((30, 100, 4))
    img_buffer[:, :50, 3] = 1.0
    img_buffer[:, 50:, 3] = 0.5
    for ii in range(3):
        img_buffer[ii * 10:(ii + 1) * 10, :, ii] = 1.0
    img = visual.RawImage(ec, img_buffer)

    # make a line
    line = visual.Line(ec, [[-2, 2, 2, -2], [-2, 2, -2, -2]], units='deg',
                       line_color='w', line_width=2.0)

    # make a rectangle
    rect = visual.Rectangle(ec, [0, 0, 2, 2], units='deg', fill_color='k')

    # make a circle
    circle = visual.Circle(ec, 1, units='deg', line_color='w', fill_color='k',
                           line_width=2.0)

    # do the drawing, then flip
    img.draw()
    line.draw()
    rect.draw()
    circle.draw()
    screenshot = ec.screenshot()  # must be called *before* the flip
    ec.flip()
    ec.wait_for_presses(0.5)

import matplotlib.pyplot as plt
plt.ion()
ea.plot_screen(screenshot)
