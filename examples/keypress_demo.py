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

print __doc__

from expyfun import ExperimentController
# from expyfun.utils import set_log_level
# set_log_level('DEBUG')

isi = 0.5
wait_dur = 3.0
msg_dur = 3.0

with ExperimentController('KeypressDemo', 'psychopy', 'keyboard', screen_num=0,
                          window_size=[640, 480], full_screen=False,
                          stim_db=0, noise_db=0, stim_fs=0,
                          participant='foo', session='001') as ec:
    ec.wait_secs(isi)

    # the screen_prompt method
    pressed = ec.screen_prompt('screen_prompt\npress any key',
                               max_wait=wait_dur, timestamp=True)
    ec.write_data_line('screen_prompt', pressed)
    if pressed[0] is None:
        message = 'no keys pressed'
    else:
        message = '{} pressed after {} secs'.format(pressed[0],
                                                    round(pressed[1], 4))
    ec.screen_prompt(message, msg_dur)
    ec.clear_screen()
    ec.wait_secs(isi)

    # the wait_for_presses method, relative to master clock
    ec.screen_text('press a few keys\n\nwait_for_presses\nmax_wait={}\n'
                   'timestamp relative to master clock'.format(wait_dur))
    pressed = ec.wait_for_presses(wait_dur, relative_to=0.0)
    ec.write_data_line('wait_for_presses relative_to 0.0', pressed)
    if not len(pressed):
        message = 'no keys pressed'
    else:
        message = ['{} pressed at {} secs'
                   ''.format(key, round(time, 4)) for key, time in pressed]
        message = '\n'.join(message)
    ec.screen_prompt(message, msg_dur)
    ec.clear_screen()
    ec.wait_secs(isi)

    # the wait_for_presses method, relative to method call
    ec.screen_text('press a few keys\n\nwait_for_presses\nmax_wait={}\n'
                   'timestamp relative to when method called'.format(wait_dur))
    pressed = ec.wait_for_presses(wait_dur, relative_to=None)
    ec.write_data_line('wait_for_presses relative_to None', pressed)
    if not len(pressed):
        message = 'no keys pressed'
    else:
        message = ['{} pressed after {} secs'
                   ''.format(key, round(time, 4)) for key, time in pressed]
        message = '\n'.join(message)
    ec.screen_prompt(message, msg_dur)
    ec.clear_screen()
    ec.wait_secs(isi)

    # the listen_presses / get_presses methods, relative to master clock
    disp_time = wait_dur
    countdown = ec.current_time + disp_time
    ec.listen_presses()
    while ec.current_time < countdown:
        cur_time = round(countdown - ec.current_time, 1)
        if cur_time != disp_time:
            disp_time = cur_time
            ec.screen_text('press a few keys\n\nlisten_presses\n{}\ntimestamp '
                           'relative to master clock'.format(disp_time))
    pressed = ec.get_presses(relative_to=0.0)
    ec.write_data_line('get_presses relative_to 0.0', pressed)
    if not len(pressed):
        message = 'no keys pressed'
    else:
        message = ['{} pressed at {} secs'
                   ''.format(key, round(time, 4)) for key, time in pressed]
        message = '\n'.join(message)
    ec.screen_prompt(message, msg_dur)
    ec.clear_screen()
    ec.wait_secs(isi)

    # the listen_presses / get_presses methods, relative to method call
    disp_time = wait_dur
    countdown = ec.current_time + disp_time
    ec.listen_presses()
    while ec.current_time < countdown:
        cur_time = round(countdown - ec.current_time, 1)
        if cur_time != disp_time:
            disp_time = cur_time
            ec.screen_text('press a few keys\n\nlisten_presses\n{}\ntimestamp '
                           'relative to when method called'.format(disp_time))
    pressed = ec.get_presses()
    ec.write_data_line('get_presses relative_to None', pressed)
    if not len(pressed):
        message = 'no keys pressed'
    else:
        message = ['{} pressed after {} secs'
                   ''.format(key, round(time, 4)) for key, time in pressed]
        message = '\n'.join(message)
    ec.screen_prompt(message, msg_dur)
