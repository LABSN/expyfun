"""
=======================
KeyPressAndRelease demo
=======================

This example demonstrates gathering key-releases as well as presses with
the ExperimentController class.

Please note that this currently only works for the keyboard, which has
inprecise timing.

.. warning:

    It is currently not possible to get key-release events for Cedrus boxes or
    TDT. Therefore, using get_presses(kind='releases') or
    get_presses(kind='both') will throw an exception.

"""
# Author: Jasper van den Bosch <jasperb@uw.edu>
#
# License: BSD (3-clause)

from expyfun import ExperimentController

print(__doc__)

isi = 0.5
wait_dur = 3.0
msg_dur = 3.0

with ExperimentController('KeyPressAndReleaseDemo', screen_num=0,
                          window_size=[1280, 960], full_screen=False,
                          stim_db=0, noise_db=0, output_dir=None,
                          participant='foo', session='001',
                          version='dev') as ec:
    ec.wait_secs(isi)

    ###########################################
    # listen_presses / while loop / get_presses(kind='both')
    instruction = ("Press and release some keys\n\nlisten_presses()"
                   "\nwhile loop {}\nget_presses(kind='both')")
    disp_time = wait_dur
    countdown = ec.current_time + disp_time
    ec.call_on_next_flip(ec.listen_presses)
    ec.screen_text(instruction.format(disp_time))
    ec.flip()
    while ec.current_time < countdown:
        cur_time = round(countdown - ec.current_time, 1)
        if cur_time != disp_time:
            disp_time = cur_time
            # redraw text with updated disp_time
            ec.screen_text(instruction.format(disp_time))
            ec.flip()
    events = ec.get_presses(kind='both')
    ec.write_data_line('listen / while / get_presses', events)
    if not len(events):
        message = 'no keys pressed'
    else:
        message = ['{} {} after {} secs\n'
                   ''.format(k, r, round(t, 4)) for k, t, r in events]
        message = ''.join(message)
    ec.screen_prompt(message, msg_dur)
    ec.wait_secs(isi)
