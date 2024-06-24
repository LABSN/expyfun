"""
============================================
Sound level test and visual size calibration
============================================

This example tests the audio level and video size. For audio, it produces an 80
db SPL 1000 Hz tone (note that at 1000 Hz, the frequency weighting for SPL
measurement shouldn't matter). For video, it produces a square that should be
10 degrees visual angle and tells you what the physical width should be in cm.
This of course depends on correct settings for monitor width, resolution, and
distance.
"""
# Author: Ross Maddox <rkmaddox@uw.edu>
#
# License: BSD (3-clause)

import numpy as np

import expyfun.analyze as ea
from expyfun import ExperimentController, building_doc
from expyfun.visual import Rectangle

print(__doc__)


with ExperimentController(
    "LevelTest",
    full_screen=True,
    noise_db=-np.inf,
    participant="s",
    session="0",
    output_dir=None,
    suppress_resamp=True,
    check_rms=None,
    stim_db=80,
    version="dev",
) as ec:
    tone = (
        0.01 * np.sqrt(2.0) * np.sin(2 * np.pi * 1000.0 * np.arange(0, 10, 1.0 / ec.fs))
    )
    assert np.allclose(np.sqrt(np.mean(tone * tone)), 0.01)
    square = Rectangle(ec, (0, 0, 10, 10), units="deg", fill_color="r")
    cm = (
        np.diff(ec._convert_units([[0, 5], [0, 5]], "deg", "pix"), axis=-1)[0]
        / ec.dpi
        / 0.39370
    )
    ec.load_buffer(tone)  # RMS == 0.01
    pressed = None
    screenshot = None
    while pressed != "8":  # enable a clean quit if required
        square.draw()
        ec.screen_text(f"Width: {np.round(2 * cm, 1)} cm", wrap=False)
        ec.screen_text(f"Output level: {ec.stim_db} dB", wrap=True)
        screenshot = ec.screenshot() if screenshot is None else screenshot
        t1 = ec.start_stimulus(start_of_trial=False)  # skip checks
        pressed = ec.wait_one_press(10)[0] if not building_doc else "8"
        ec.flip()
        ec.wait_one_press(0.5 if not building_doc else 0.0)
        ec.stop()

ea.plot_screen(screenshot)
