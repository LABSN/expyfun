"""
===========================
Run a very basic experiment
===========================

This example demonstrates an (almost) minimum working example of the
ExperimentController class.
"""
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

from expyfun import ExperimentController, analyze, building_doc
from expyfun.visual import FixationDot

print(__doc__)

# set configuration
fs = 24414.0  # default for ExperimentController
dur = 1.0
tone = np.sin(2 * np.pi * 1000 * np.arange(int(fs * dur)) / float(fs))
tone *= 0.01 * np.sqrt(2)  # Set RMS to 0.01
max_wait = 1.0 if not building_doc else 0.0

with ExperimentController(
    "testExp", participant="foo", session="001", output_dir=None, version="dev"
) as ec:
    ec.screen_prompt("Press a button when you hear the tone", max_wait=max_wait)

    dot = FixationDot(ec)
    ec.load_buffer(tone)
    dot.draw()
    screenshot = ec.screenshot()  # only because we want to show it in the docs

    ec.identify_trial(ec_id="tone", ttl_id=[0, 0])
    ec.start_stimulus()
    presses = ec.wait_for_presses(dur if not building_doc else 0.0)
    ec.trial_ok()
    print(f"Presses:\n{presses}")

analyze.plot_screen(screenshot)
