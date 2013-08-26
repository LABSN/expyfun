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

from expyfun import ExperimentController, EyelinkController
from psychopy import visual
import numpy as np

link = '100.1.1.1'  # or 'dummy' for fake operation


with ExperimentController('testExp', full_screen=True,
                          participant='foo', session='001') as ec:
    el = EyelinkController(ec, link=link)
    ec.screen_prompt('Welcome to the experiment!\n\nFirst, we will '
                     'perform a screen calibration.\n\nPress a button '
                     'to continue.')
    el.calibrate()  # by default this starts recording EyeLink data
    ec.screen_prompt('Excellent! Now, follow the red circle around the edge '
                     'of the big white circle.\n\nPress a button to continue')
    ec.clear_screen()

    # make some circles to be drawn
    radius = 10  # in degrees
    theta = np.linspace(np.pi / 2., 2.5 * np.pi, 200)
    x_pos, y_pos = radius * np.cos(theta), radius * np.sin(theta)
    big_circ = visual.Circle(ec.window, 10, edges=100, units='deg')
    targ_circ = visual.Circle(ec.window, 0.2, edges=100, units='deg',
                              fillColor=(1., -1., -1.), lineColor=None)
    targ_circ.setPos((x_pos[0], y_pos[0]), units='deg', log=False)

    el.stamp_trial_id(1)  # stamp this trial ID to the file
    # now let's make it so the SYNC message gets stamped when we flip
    ec.call_on_next_flip(el.stamp_trial_start)
    fix_pos = ec.deg2pix((x_pos[0], y_pos[0]))

    # start out by waiting for a 1 sec fixation at the start
    big_circ.draw()
    targ_circ.draw()
    ec.flip()
    if not el.wait_for_fix(fix_pos, 1., max_wait=5.):
        print 'Initial fixation failed'
    for ii, (x, y) in enumerate(zip(x_pos[1:], y_pos[1:])):
        targ_circ.setPos((x, y), units='deg', log=False)
        big_circ.draw()
        targ_circ.draw()
        ec.flip()
        fix_pos = ec.deg2pix((x, y))
        if not el.wait_for_fix(fix_pos, max_wait=5.):
            print 'Fixation {0} failed'.format(ii + 1)
    # eyelink auto-closes (el.close()) because it gets registered with EC
