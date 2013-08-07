from os import path as op
import numpy as np
from scipy import io as sio
from psychopy import core

from expyfun import ExperimentController
from generate_stimuli import generate_stimuli

# set configuration
noise_amp = 45  # dB for background noise
stim_amp = 75  # dB for stimuli
min_resp_time = 0.1
max_resp_time = 1.0
feedback_dur = 1.5
isi = 0.2
running_total = 0

# if the stimuli have not been made, let's make them in examples dir
stimulus_dir = op.split(__file__)[0]
stimulus_file = op.join(stimulus_dir, 'equally_spaced_sinewaves.mat')
if not op.isfile(stimulus_file):
    generate_stimuli(output_dir=stimulus_dir)

# load stimuli (from a call to generate_stimuli() from generate_stimuli.py)
stims = sio.loadmat('equally_spaced_sinewaves.mat')
orig_rms = stims['rms'][0]
freqs = stims['freqs'][0]
fs = stims['fs'][0][0]
trial_order = stims['trial_order'][0]
num_trials = len(trial_order)
num_freqs = len(freqs)

if num_freqs > 8:
    raise RuntimeError('Too many frequencies, not enough buttons.')

# keep only sinusoid data, convert dictionary to list of arrays, make stereo
wavs = {k: np.c_[stims[k], stims[k]] for k in stims if 'stim_' in k}
wavs = [v for k, v in sorted(wavs.items())]

# instructions
instructions = ('You will hear tones at {0} different frequencies. Your job is'
                ' to press the button corresponding to that frequency. Please '
                'press buttons 1-{0} now to hear each tone.').format(num_freqs)

instr_finished = ('Okay, now press any of those buttons to start the real '
                  'thing.')

with ExperimentController('testExp', 'psychopy', 'keyboard', stim_amp=75,
                          noise_amp=45) as ec:
    # define usable buttons / keys
    live_keys = [x + 1 for x in range(num_freqs)]
    not_yet_pressed = [x + 1 for x in range(num_freqs)]

    # show instructions until all buttons have been pressed at least once
    ec.init_trial()
    ec.screen_prompt(instructions, max_wait=0)
    while len(not_yet_pressed) > 0:
        print 'hello'
        pressed = ec.get_buttons(live_keys)  
        # don't need to use wait_buttons here because the waiting is
        # encoded by the while loop
        for p in pressed:
            p = int(p)
            ec.load_buffer(wavs[p - 1])
            ec.flip_and_play()
            ec.wait_secs(len(wavs[p - 1]) / ec.fs)
            if p in not_yet_pressed:
                not_yet_pressed.pop(not_yet_pressed.index(p))
    ec.clear_buffer()
    ec.clear_screen()
    ec.wait_secs(isi)

    # show instructions finished screen #
    ec.screen_prompt(instr_finished)
    ec.check_force_quit()
    ec.clear_screen()
    ec.wait_secs(isi)

    # run trials
    ec.screen_prompt('OK, here we go!', max_wait=feedback_dur)
    ec.clear_screen()
    ec.wait_secs(isi)
    for stim_num in trial_order:
        ec.init_trial()
        ec.clear_buffer()
        ec.load_buffer(wavs[stim_num])
        ec.flip_and_play()
        button, rt = ec.wait_buttons()
        ec.stop_reset()
        ec.check_force_quit()
        ec.save_button_presses()
        # some feedback
        if int(button) == stim_num + 1:
            running_total = running_total + 1
            text = 'Correct! Your reaction time was %0.3f' % rt
        else:
            text = ('You pressed %i, the correct answer was %i'
                    (pressed[0], stim_num + 1))
        ec.screen_prompt(text, max_wait=feedback_dur)
        ec.clear_screen()
        ec.wait_secs(isi)
        
    # # # # # # # # # #
    # end experiment  #
    # # # # # # # # # #
    ec.screen_prompt('All done! You got {0} correct out of {1} '
                     'trials.'.format(running_total, num_trials))
    ec.wait_secs(feedback_dur)
    ec.clear_screen()
