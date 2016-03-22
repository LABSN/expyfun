"""
=============
Keyrelease demo
=============

This example demonstrates the keyrelease-gathering technique available
in the ExperimentController class.
"""
# Author: Dan McCloy <drmccloy@uw.edu>
#
# License: BSD (3-clause)

from expyfun import ExperimentController

print(__doc__)

isi = 0.5
wait_dur = 3.0
msg_dur = 3.0

with ExperimentController('KeyreleaseDemo', screen_num=0,
                          window_size=[1280, 960], full_screen=False,
                          stim_db=0, noise_db=0, output_dir=None,
                          participant='foo', session='001',
                          version='dev') as ec:
    ec.wait_secs(isi)

    ###########################################
    # listen_releases / while loop / get_releases
    disp_time = wait_dur
    countdown = ec.current_time + disp_time
    ec.call_on_next_flip(ec.listen_presses)
    ec.screen_text('release some keys\n\nlisten_presses()'
                   '\nwhile loop {}\nget_releases()'.format(disp_time))
    ec.flip()
    while ec.current_time < countdown:
        cur_time = round(countdown - ec.current_time, 1)
        if cur_time != disp_time:
            disp_time = cur_time
            # redraw text with updated disp_time
            ec.screen_text('release some keys\n\nlisten_presses() '
                           '\nwhile loop {}\nget_releases()'.format(disp_time))
            ec.flip()
    released = ec.get_releases()
    ec.write_data_line('listen / while / get_releases', released)
    if not len(released):
        message = 'no keys released'
    else:
        message = ['{} released after {} secs\n'
                   ''.format(key, round(time, 4)) for key, time in released]
        message = ''.join(message)
    ec.screen_prompt(message, msg_dur)
    ec.wait_secs(isi)
