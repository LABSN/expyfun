"""
=============
Keypress demo
=============

This example demonstrates the different keypress-gathering techniques available
in the ExperimentController class.
"""
# Author: Dan McCloy <drmccloy@uw.edu>
#
# License: BSD (3-clause)

import expyfun.analyze as ea
from expyfun import ExperimentController, building_doc

print(__doc__)


isi = 0.5
wait_dur = 3.0 if not building_doc else 0.0
msg_dur = 3.0 if not building_doc else 0.0

with ExperimentController(
    "KeypressDemo",
    screen_num=0,
    window_size=[640, 480],
    full_screen=False,
    stim_db=0,
    noise_db=0,
    output_dir=None,
    participant="foo",
    session="001",
    version="dev",
) as ec:
    ec.wait_secs(isi)

    ###############
    # screen_prompt
    pressed = ec.screen_prompt(
        "press any key\n\nscreen_prompt(" f"max_wait={wait_dur})",
        max_wait=wait_dur,
        timestamp=True,
    )
    ec.write_data_line("screen_prompt", pressed)
    if pressed[0] is None:
        message = "no keys pressed"
    else:
        message = f"{pressed[0]} pressed after {round(pressed[1], 4)} secs"
    ec.screen_prompt(message, msg_dur)
    ec.wait_secs(isi)

    ##################
    # wait_for_presses
    ec.screen_text(f"press some keys\n\nwait_for_presses(max_wait={wait_dur})" "")
    screenshot = ec.screenshot()
    ec.flip()
    pressed = ec.wait_for_presses(wait_dur)
    ec.write_data_line("wait_for_presses", pressed)
    if not len(pressed):
        message = "no keys pressed"
    else:
        message = [
            f"{key} pressed after {round(time, 4)} secs\n" "" for key, time in pressed
        ]
        message = "".join(message)
    ec.screen_prompt(message, msg_dur)
    ec.wait_secs(isi)

    ############################################
    # wait_for_presses, relative to master clock
    ec.screen_text(
        f"press some keys\n\nwait_for_presses(max_wait={wait_dur}, " "relative_to=0.0)"
    )
    ec.flip()
    pressed = ec.wait_for_presses(wait_dur, relative_to=0.0)
    ec.write_data_line("wait_for_presses relative_to 0.0", pressed)
    if not len(pressed):
        message = "no keys pressed"
    else:
        message = [
            f"{key} pressed at {round(time, 4)} secs\n" "" for key, time in pressed
        ]
        message = "".join(message)
    ec.screen_prompt(message, msg_dur)
    ec.wait_secs(isi)

    ##########################################
    # listen_presses / wait_secs / get_presses
    ec.screen_text(
        f"press some keys\n\nlisten_presses()\nwait_secs({wait_dur})" "\nget_presses()"
    )
    ec.flip()
    ec.listen_presses()
    ec.wait_secs(wait_dur)
    pressed = ec.get_presses()  # relative_to=0.0
    ec.write_data_line("listen / wait / get_presses", pressed)
    if not len(pressed):
        message = "no keys pressed"
    else:
        message = [
            f"{key} pressed after {round(time, 4)} secs\n" "" for key, time in pressed
        ]
        message = "".join(message)
    ec.screen_prompt(message, msg_dur)
    ec.wait_secs(isi)

    ####################################################################
    # listen_presses / wait_secs / get_presses, relative to master clock
    ec.screen_text(
        "press a few keys\n\nlisten_presses()"
        f"\nwait_secs({wait_dur})\nget_presses(relative_to=0.0)"
        ""
    )
    ec.flip()
    ec.listen_presses()
    ec.wait_secs(wait_dur)
    pressed = ec.get_presses(relative_to=0.0)
    ec.write_data_line("listen / wait / get_presses relative_to 0.0", pressed)
    if not len(pressed):
        message = "no keys pressed"
    else:
        message = [
            f"{key} pressed at {round(time, 4)} secs\n" "" for key, time in pressed
        ]
        message = "".join(message)
    ec.screen_prompt(message, msg_dur)
    ec.wait_secs(isi)

    ###########################################
    # listen_presses / while loop / get_presses
    disp_time = wait_dur
    countdown = ec.current_time + disp_time
    ec.call_on_next_flip(ec.listen_presses)
    ec.screen_text(
        "press some keys\n\nlisten_presses()" f"\nwhile loop {disp_time}\nget_presses()"
    )
    ec.flip()
    while ec.current_time < countdown:
        cur_time = round(countdown - ec.current_time, 1)
        if cur_time != disp_time:
            disp_time = cur_time
            # redraw text with updated disp_time
            ec.screen_text(
                "press some keys\n\nlisten_presses() "
                f"\nwhile loop {disp_time}\nget_presses()"
            )
            ec.flip()
    pressed = ec.get_presses()
    ec.write_data_line("listen / while / get_presses", pressed)
    if not len(pressed):
        message = "no keys pressed"
    else:
        message = [
            f"{key} pressed after {round(time, 4)} secs\n" "" for key, time in pressed
        ]
        message = "".join(message)
    ec.screen_prompt(message, msg_dur)
    ec.wait_secs(isi)

    #####################################################################
    # listen_presses / while loop / get_presses, relative to master clock
    disp_time = wait_dur
    countdown = ec.current_time + disp_time
    ec.call_on_next_flip(ec.listen_presses)
    ec.screen_text(
        "press some keys\n\nlisten_presses()\nwhile loop "
        f"{disp_time}\nget_presses(relative_to=0.0)"
    )
    ec.flip()
    while ec.current_time < countdown:
        cur_time = round(countdown - ec.current_time, 1)
        if cur_time != disp_time:
            disp_time = cur_time
            # redraw text with updated disp_time
            ec.screen_text(
                "press some keys\n\nlisten_presses()\nwhile "
                f"loop {disp_time}\nget_presses(relative_to=0.0)"
                ""
            )
            ec.flip()
    pressed = ec.get_presses(relative_to=0.0)
    ec.write_data_line("listen / while / get_presses relative_to 0.0", pressed)
    if not len(pressed):
        message = "no keys pressed"
    else:
        message = [
            f"{key} pressed at {round(time, 4)} secs\n" "" for key, time in pressed
        ]
        message = "".join(message)
    ec.screen_prompt(message, msg_dur)

ea.plot_screen(screenshot)
