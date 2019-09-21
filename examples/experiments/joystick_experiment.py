"""
=====================
Use joystick controls
=====================

This example demonstrates how to use a joystick as an input device.
"""
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

# Todo:
# - make it actually work (Pyglet bug)
# - make it move around the screen

from expyfun import ExperimentController, analyze, building_doc
from expyfun.visual import Circle, Text

print(__doc__)

joystick = not building_doc
with ExperimentController('joyExp', participant='foo', session='001',
                          output_dir=None, version='dev',
                          joystick=joystick) as ec:
    circle = Circle(ec, 1, units='deg', fill_color='k', line_color='w')
    pressed = ''
    if not building_doc:
        ec.listen_joystick_button_presses()
    count = 0
    screenshot = None
    while pressed != '2':  # enable a clean quit if required
        Text(ec, str(count), pos=(1, -1),
             anchor_x='right', anchor_y='bottom').draw()
        circle.draw()
        ec.flip()
        screenshot = ec.screenshot() if screenshot is None else screenshot
        if not building_doc:
            pressed = ec.get_joystick_button_presses()
        else:
            pressed = [('2',)]
        pressed = pressed[0][0] if pressed else ''
        count += 1
        ec.wait_secs(0.2)
        ec.check_force_quit()

analyze.plot_screen(screenshot)
