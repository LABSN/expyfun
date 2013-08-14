from os import path as op
import numpy as np
from scipy import io as sio

from expyfun import ExperimentController
from generate_stimuli import generate_stimuli

# set configuration
noise_amp = 45  # dB for background noise
stim_amp = 75  # dB for stimuli
min_resp_time = 0.1
max_resp_time = 2.0
feedback_dur = 0.5
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
num_trials = len(trial_order) / 2 + 1  # 1/2 of stims presented as mass trial
num_freqs = len(freqs)

if num_freqs > 8:
    raise RuntimeError('Too many frequencies, not enough buttons.')

# keep only sinusoids, make stereo, order low - high, convert to list of arrays
wavs = {k: np.r_[stims[k], stims[k]].T for k in stims if 'stim_' in k}
wavs = [v for k, v in sorted(wavs.items())]
# instructions
instructions = ('You will hear tones at {0} different frequencies. Your job is'
                ' to press the button corresponding to that frequency. Please '
                'press buttons 1-{0} now to hear each tone.').format(num_freqs)

instr_finished = ('Okay, now press any of those buttons to start the real '
                  'thing.')

with ExperimentController('testExp', 'psychopy', 'keyboard', screen_num=0,
                          window_size=[800, 600], full_screen=False,
                          stim_amp=75, noise_amp=45, participant='foo',
                          session='001') as ec:
    ec.set_noise_amp(45)
    ec.set_stim_amp(75)
    # define usable buttons / keys
    live_keys = [x + 1 for x in range(num_freqs)]
    not_yet_pressed = live_keys[:]

    ec.init_trial()  # resets trial clock, clears keyboard buffer, etc
    ec.screen_text(instructions)
    # show instructions until all buttons have been pressed at least once
    ec.add_data_line({'stim_num': 'training'})
    while len(not_yet_pressed) > 0:
        (pressed, timestamp) = ec.get_press(live_keys=live_keys)
        for p in pressed:
            p = int(p)
            ec.load_buffer(wavs[p - 1])
            ec.flip_and_play()
            ec.wait_secs(len(wavs[p - 1]) / float(ec.fs))
            ec.stop_reset()
            if p in not_yet_pressed:
                not_yet_pressed.pop(not_yet_pressed.index(p))
    ec.clear_buffer()
    ec.clear_screen()
    ec.wait_secs(isi)

    # show instructions finished screen
    ec.add_data_line({'stim_num': 'prompt'})
    ec.screen_prompt(instr_finished, live_keys=live_keys)
    ec.clear_screen()
    ec.wait_secs(isi)

    ec.screen_prompt('OK, here we go!', max_wait=feedback_dur, live_keys=None)
    ec.clear_screen()
    ec.wait_secs(isi)

    single_trials = trial_order[range(int(len(trial_order) / 2))]
    mass_trial = trial_order[int(len(trial_order) / 2):]
    # run half the trials
    for stim_num in single_trials:
        ec.add_data_line({'stim_num': stim_num})
        ec.clear_buffer()
        ec.load_buffer(wavs[stim_num])
        ec.init_trial()
        ec.flip_and_play()
        (pressed, timestamp) = ec.get_press(max_resp_time, min_resp_time,
                                            live_keys)
        ec.stop_reset()  # will stop stim playback as soon as response logged
        # some feedback
        if int(pressed) == stim_num + 1:
            running_total += 1
            message = ('Correct! Your reaction time was '
                       '{}').format(round(timestamp, 3))
        else:
            message = ('You pressed {0}, the correct answer was '
                       '{1}.').format(pressed, stim_num + 1)
        ec.screen_prompt(message, max_wait=feedback_dur)  # live_keys=live_keys
        ec.clear_screen()
        ec.wait_secs(isi)

    # add 250 ms pause between stims
    pause = np.zeros((int(ec.fs) / 4, 2))
    stims = [np.row_stack((wavs[stim], pause)) for stim in mass_trial]
    # run mass trial
    mass_stim = np.row_stack((stims[stim_num] for stim_num in mass_trial))
    ec.screen_prompt('Now you will hear {0} tones in a row. After they stop, '
                     'you will have {1} seconds to push the buttons in the '
                     'order that the tones played in. Press one of the buttons'
                     ' to begin.'.format(len(mass_trial), max_resp_time),
                     live_keys=live_keys)
    ec.clear_screen()
    ec.clear_buffer()
    ec.load_buffer(mass_stim)
    ec.add_data_line({'stim_num': mass_trial})
    ec.init_trial()
    ec.flip_and_play()
    ec.wait_secs(len(mass_stim) / float(ec.fs))
    ec.screen_text('Go!')
    presses = ec.get_presses(max_resp_time, min_resp_time, live_keys)
    print presses
    correct = [presses[n] == map(str, mass_trial)[n] for n in
               range(len(mass_trial))]
    if all(correct):
        running_total += 1
    ec.screen_prompt('You got {} out of {} correct.'.format(sum(correct),
                     len(correct)), max_wait=feedback_dur)

    # end experiment
    ec.add_data_line({'stim_num': 'prompt'})
    ec.screen_prompt('All done! You got {0} correct out of {1} trials. Press '
                     '{2} to close.'.format(running_total, num_trials,
                                            ec._force_quit[0]))
    ec.clear_screen()
