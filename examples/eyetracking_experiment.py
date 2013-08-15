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

from expyfun import ExperimentController, EyelinkController, wait_secs


with ExperimentController('testExp', full_screen=True,
                          participant='foo', session='001') as ec:
    el = EyelinkController(ec)

    ec.init_trial()  # resets trial clock, clears keyboard buffer, etc
    ec.screen_text('Hello!')
    wait_secs(1.0)
    ec.clear_screen()

    # XXX Demo calibration
    # XXX Demo drawing cursor wherever eye is
    # XXX Demo requiring holding fixation
    ec.close()
