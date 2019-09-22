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
move_rate = 0.05
noise_thresh = 0.01
# on a Logitech Cordless Rumblepad, the right stick is the analog one,
# and it has values stored in z and rz
joy_keys = ('z', 'rz')
with ExperimentController('joyExp', participant='foo', session='001',
                          output_dir=None, version='dev',
                          joystick=joystick) as ec:
    circles = [Circle(ec, 0.5, units='deg', fill_color='none', line_color='w')]
    # We use normalized units for "pos" so we need to compensate in movement
    # so that X/Y movement is even
    ratios = [1., ec.window_size_pix[0] / float(ec.window_size_pix[1])]
    pressed = ''
    if not building_doc:
        ec.listen_joystick_button_presses()
    count = 0
    screenshot = None
    pos = [0., 0.]
    while pressed != '2':  # enable a clean quit if required
        #######################################################################
        # Draw things

        Text(ec, str(count), pos=(1, -1),
             anchor_x='right', anchor_y='bottom').draw()
        for circle in circles[::-1]:
            circle.draw()
        screenshot = ec.screenshot() if screenshot is None else screenshot
        ec.flip()

        #######################################################################
        # Get presses
        if not building_doc:
            pressed = ec.get_joystick_button_presses()
        else:
            pressed = [('2',)]
        count += len(pressed)
        ec.listen_joystick_button_presses()  # clear events

        #######################################################################
        # Move the cursor
        for idx, (key, ratio) in enumerate(zip(joy_keys, ratios)):
            delta = ec.get_joy_value(key)
            if abs(delta) > noise_thresh:  # remove noise
                pos[idx] = max(min(
                    pos[idx] + move_rate * ratio * delta, 1), -1)
        circles[0].set_pos(pos, units='norm')
        if pressed:
            circles.insert(1, Circle(ec, 1, units='deg',
                           fill_color='r', line_color='w'))
            circles[1].set_pos(pos, units='norm')
            if len(circles) > 5:
                circles.pop(-1)
        ec.check_force_quit()

analyze.plot_screen(screenshot)
