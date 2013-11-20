"""
============================
Experiment with eye-tracking
============================

Integration with Eyelink functionality makes programming experiments
using eye-tracking simpler.
"""
# Author: Eric Larson <larsoner@uw.edu>
#
# License: BSD (3-clause)

print __doc__

import numpy as np

from expyfun import ExperimentController, EyelinkController, visual
import expyfun.analyze as ea

link = None  # or '100.1.1.1' for real eye tracking


with ExperimentController('testExp', full_screen=True, participant='foo',
                          session='001', output_dir=None) as ec:
    el = EyelinkController(ec, link=link)
    ec.screen_prompt('Welcome to the experiment!<br><br>First, we will '
                     'perform a screen calibration.<br><br>Press a button '
                     'to continue.')
    el.calibrate()  # by default this starts recording EyeLink data
    ec.screen_prompt('Excellent! Now, follow the red circle around the edge '
                     'of the big white circle.<br><br>Press a button to '
                     'continue')

    # make some circles to be drawn
    radius = 7.5  # degrees
    targ_rad = 0.2  # degrees
    theta = np.linspace(np.pi / 2., 2.5 * np.pi, 200)
    x_pos, y_pos = radius * np.cos(theta), radius * np.sin(theta)
    big_circ = visual.Circle(ec, radius, (0, 0), units='deg',
                             fill_color=None, line_color='white',
                             line_width=3.0)
    targ_circ = visual.Circle(ec, targ_rad, (x_pos[0], y_pos[0]),
                              units='deg', fill_color='red')

    el.stamp_trial_id(1)  # stamp this trial ID to the file
    # now let's make it so the SYNC message gets stamped when we flip
    ec.call_on_next_flip(el.stamp_trial_start)
    fix_pos = (x_pos[0], y_pos[0])

    # start out by waiting for a 1 sec fixation at the start
    big_circ.draw()
    targ_circ.draw()
    screenshot = ec.screenshot()
    ec.flip()
    if not el.wait_for_fix(fix_pos, 1., max_wait=5., units='deg'):
        print 'Initial fixation failed'
    for ii, (x, y) in enumerate(zip(x_pos[1:], y_pos[1:])):
        targ_circ.set_pos((x, y), units='deg')
        big_circ.draw()
        targ_circ.draw()
        ec.flip()
        if not el.wait_for_fix([x, y], max_wait=5., units='deg'):
            print 'Fixation {0} failed'.format(ii + 1)
    el.stop()  # stop recording to save the file
    ec.screen_prompt('All done!', max_wait=1.0)
    # eyelink auto-closes (el.close()) because it gets registered with EC

import matplotlib.pyplot as plt
plt.ion()
ea.plot_screen(screenshot)
