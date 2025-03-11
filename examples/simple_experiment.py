"""
=======================
Run a simple experiment
=======================

This example demonstrates much of the basic functionality built into
the ExperimentController class.
"""
# Author: Dan McCloy <drmccloy@uw.edu>
#
# License: BSD (3-clause)

import os
import sys
from os import path as op

import numpy as np

import expyfun.analyze as ea
from expyfun import (
    ExperimentController,
    building_doc,
    get_keyboard_input,
    set_log_level,
)
from expyfun.io import read_hdf5

print(__doc__)

set_log_level("INFO")

# set configuration
noise_db = 45  # dB for background noise
stim_db = 65  # dB for stimuli
min_resp_time = 0.1
max_resp_time = 2.0
max_wait = np.inf
feedback_dur = 2.0
isi = 0.2
running_total = 0

# make the stimuli if necessary and then load them
fname = "equally_spaced_sinewaves.hdf5"
if not op.isfile(fname):
    # This sys.path wrangling is only necessary for Sphinx automatic
    # documentation building
    sys.path.insert(0, os.getcwd())
    from generate_simple_stimuli import generate_stimuli

    generate_stimuli()
stims = read_hdf5(fname)
orig_rms = stims["rms"]
freqs = stims["freqs"]
fs = stims["fs"]
trial_order = stims["trial_order"]
num_trials = len(trial_order)
num_freqs = len(freqs)

if num_freqs > 8:
    raise RuntimeError("Too many frequencies, not enough buttons.")

# keep only sinusoids, order low-high, convert to list of arrays
wavs = [stims[k] for k in sorted(stims.keys()) if k.startswith("stim_")]

# instructions
instructions = (
    f"You will hear tones at {num_freqs} different frequencies. Your job is"
    " to press the button corresponding to that frequency. Please "
    f"press buttons 1-{num_freqs} now to hear each tone."
)

instr_finished = (
    "Okay, now press any of those buttons to start the real "
    "thing. There will be background noise."
)

with ExperimentController(
    "testExp",
    verbose=True,
    screen_num=0,
    window_size=[800, 600],
    full_screen=False,
    stim_db=stim_db,
    noise_db=noise_db,
    stim_fs=fs,
    participant="foo",
    session="001",
    version="dev",
    output_dir=None,
) as ec:
    # define usable buttons / keys
    live_keys = [x + 1 for x in range(num_freqs)]

    # do training, or not
    ec.set_visible(False)
    long_resp_time = max_resp_time + 1
    if building_doc:
        max_wait = max_resp_time = min_resp_time = train = feedback_dur = 0
        long_resp_time = 0
    else:
        train = get_keyboard_input("Run training (0=no, 1=yes [default]): ", 1, int)
    ec.set_visible(True)

    if train:
        not_yet_pressed = live_keys[:]

        # show instructions until all buttons have been pressed at least once
        ec.screen_text(instructions)
        ec.flip()
        while len(not_yet_pressed) > 0:
            pressed, timestamp = ec.wait_one_press(live_keys=live_keys)
            for p in pressed:
                p = int(p)
                ec.load_buffer(wavs[p - 1])
                ec.play()
                ec.wait_secs(len(wavs[p - 1]) / float(ec.fs))
                ec.stop()
                if p in not_yet_pressed:
                    not_yet_pressed.pop(not_yet_pressed.index(p))
        ec.flip()  # clears the screen
        ec.wait_secs(isi)

    # show instructions finished screen
    ec.screen_prompt(instr_finished, live_keys=live_keys, max_wait=max_wait)
    ec.wait_secs(isi)

    ec.call_on_next_flip(ec.start_noise())
    ec.screen_text("OK, here we go!", wrap=False)
    screenshot = ec.screenshot()
    ec.wait_one_press(max_wait=feedback_dur, live_keys=None)
    ec.wait_secs(isi)

    single_trial_order = trial_order[range(len(trial_order) // 2)]
    mass_trial_order = trial_order[len(trial_order) // 2 :]
    # run the single-tone trials
    for stim_num in single_trial_order:
        ec.load_buffer(wavs[stim_num])
        ec.identify_trial(ec_id=stim_num, ttl_id=[0, 0])
        ec.write_data_line("one-tone trial", stim_num + 1)
        ec.start_stimulus()
        pressed, timestamp = ec.wait_one_press(max_resp_time, min_resp_time, live_keys)
        ec.stop()  # will stop stim playback as soon as response logged
        ec.trial_ok()

        # some feedback
        if pressed is None:
            message = "Too slow!"
        elif int(pressed) == stim_num + 1:
            running_total += 1
            message = f"Correct! Your reaction time was {round(timestamp, 3)}"
        else:
            message = f"You pressed {pressed}, the correct answer was {stim_num + 1}."
        ec.screen_prompt(message, max_wait=feedback_dur)
        ec.wait_secs(isi)

    # create 100 ms pause to play between stims and concatenate
    pause = np.zeros(int(ec.fs / 10))
    concat_wavs = wavs[mass_trial_order[0]]
    for num in mass_trial_order[1 : len(mass_trial_order)]:
        concat_wavs = np.r_[concat_wavs, pause, wavs[num]]
    concat_dur = len(concat_wavs) / float(ec.fs)
    # run mass trial
    ec.screen_prompt(
        f"Now you will hear {len(mass_trial_order)} tones in a row. After they stop, "
        f'wait for the "Go!" prompt, then you will have {max_resp_time} '
        "seconds to push the buttons in the order that the tones "
        "played in. Press one of the buttons to begin."
        "",
        live_keys=live_keys,
        max_wait=max_wait,
    )
    ec.load_buffer(concat_wavs)
    ec.identify_trial(ec_id="multi-tone", ttl_id=[0, 1])
    ec.write_data_line("multi-tone trial", [x + 1 for x in mass_trial_order])
    ec.start_stimulus()
    ec.wait_secs(len(concat_wavs) / float(ec.stim_fs) if not building_doc else 0)
    ec.screen_text("Go!", wrap=False)
    ec.flip()
    pressed = ec.wait_for_presses(long_resp_time, min_resp_time, live_keys, False)
    answers = [str(x + 1) for x in mass_trial_order]
    correct = [press == ans for press, ans in zip(pressed, answers)]
    running_total += sum(correct)
    ec.call_on_next_flip(ec.stop_noise())
    ec.screen_prompt(
        f"You got {sum(correct)} out of {len(answers)} correct.",
        max_wait=feedback_dur,
    )
    ec.trial_ok()

    # end experiment
    ec.screen_prompt(
        f"All done! You got {running_total} correct out of {num_trials} tones. Press "
        "any key to close.",
        max_wait=max_wait,
    )

ea.plot_screen(screenshot)
