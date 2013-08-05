from os import path as op
import numpy as np
from scipy import io as sio
from psychopy import core, event
from psychopy.constants import NOT_STARTED, STARTED, FINISHED

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

core.checkPygletDuringWait = False

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
wavs = {k: stims[k] for k in stims if k not in ('rms', 'fs', 'freqs',
                                                'trial_order', '__header__',
                                                '__globals__', '__version__')}
wavs = [np.asarray(np.column_stack((val.T, val.T)), order='C') for key, val in
        sorted(wavs.items())]

# instructions
instructions = ('You will hear tones at {0} different frequencies. Your job is'
                ' to press the button corresponding to that frequency. Please '
                'press buttons 1-{0} now to hear each tone.').format(num_freqs)

instr_finished = ('Okay, now press any of those buttons to start the real '
                  'thing.')


with ExperimentController('testExp', 'psychopy', 'keyboard', stim_amp=75,
                          noise_amp=45) as ec:
    # define usable buttons / keys
    live_keys = map(str, [x + 1 for x in range(num_freqs)])
    not_yet_pressed = live_keys[:]

    # # # # # # # # # # # # # #
    # run instructions block  #
    # # # # # # # # # # # # # #
    ec.init_trial()
    continue_trial = True

    # show instructions until all buttons have been pressed at least once
    while continue_trial:
        # TODO: wrap this block as ec.init_trial() ?
        ec.t = ec.trial_clock.getTime()
        ec.f = ec.f + 1
        if ec.t >= 0.0 and ec.button_handler.status == NOT_STARTED:
            ec.button_handler.tStart = ec.t
            ec.button_handler.frameNStart = ec.f
            ec.button_handler.status = STARTED
            ec.button_handler.clock.reset()
            event.clearEvents()

        if ec.button_handler.status == STARTED:
            pressed = event.getKeys(live_keys)
            for p in pressed:
                # normally this would be preloaded outside the while loop:
                ec.load_buffer(wavs[int(p) - 1])
                # normally this would come in its own block below
                ec.flip_and_play()
                ec.wait_secs(len(wavs[int(p) - 1]) / ec.fs)
                try:
                    del not_yet_pressed[not_yet_pressed.index(p)]
                except ValueError:
                    pass
                if len(not_yet_pressed) == 0:
                    ec.clear_buffer()
                    continue_trial = False

        # show screen prompt
        if ec.t >= 0.0 and ec.text_stim.status == NOT_STARTED:
            ec.screen_prompt(instructions)

        # try to end trial, but check if we're really done
        if not continue_trial:
            break
        continue_trial = False
        for comp in ec.trial_components:
            if hasattr(comp, 'status') and comp.status != FINISHED:
                continue_trial = True
                break  # at least one trial component not finished

        # check for force-quit
        if event.getKeys(['escape']):
            core.quit()

        # screen flip
        if continue_trial:
            ec.win.flip()

    # instructions trial is over
    ec.clear_screen()
    ec.wait_secs(isi)

    # # # # # # # # # # # # # # # # # # #
    # show instructions finished screen #
    # # # # # # # # # # # # # # # # # # #
    ec.init_trial()
    continue_trial = True

    while continue_trial:
        # TODO: wrap this block as ec.init_trial() ?
        ec.t = ec.trial_clock.getTime()
        ec.f = ec.f + 1
        if ec.t >= 0.0 and ec.button_handler.status == NOT_STARTED:
            ec.button_handler.tStart = ec.t
            ec.button_handler.frameNStart = ec.f
            ec.button_handler.status = STARTED
            ec.button_handler.clock.reset()
            event.clearEvents()

        if ec.button_handler.status == STARTED:
            pressed = event.getKeys(live_keys)
            if len(pressed) > 0:
                continue_trial = False

        # show text if necessary
        if ec.t >= 0.0 and ec.text_stim.status == NOT_STARTED:
            ec.screen_prompt(instr_finished)

        # try to end trial, but check if we're really done
        if not continue_trial:
            break
        continue_trial = False
        for comp in ec.trial_components:
            if hasattr(comp, 'status') and comp.status != FINISHED:
                continue_trial = True
                break  # at least one trial component not finished

        # check for force-quit
        if event.getKeys(['escape']):
            core.quit()

        # screen flip
        if continue_trial:
            ec.win.flip()

    # instr_finished trial is over
    ec.clear_screen()
    ec.wait_secs(0.5)

    # # # # # # # # # # #
    # run trials block  #
    # # # # # # # # # # #
    ec.screen_prompt('OK, here we go!')
    ec.wait_secs(feedback_dur)
    ec.clear_screen()
    ec.wait_secs(isi)

    for n in range(num_trials):
        ec.init_trial()
        ec.load_buffer(wavs[trial_order[n]])
        continue_trial = True

        while continue_trial:
            # TODO: wrap this block as ec.init_trial() ?
            ec.t = ec.trial_clock.getTime()
            ec.f = ec.f + 1
            if ec.t >= 0.0 and ec.button_handler.status == NOT_STARTED:
                ec.button_handler.tStart = ec.t
                ec.button_handler.frameNStart = ec.f
                ec.button_handler.status = STARTED
                ec.button_handler.clock.reset()
                event.clearEvents()

            # start audio playback
            if ec.t >= 0.0 and ec.audio.status == NOT_STARTED:
                ec.flip_and_play()

            if ec.button_handler.status == STARTED:
                pressed = event.getKeys(live_keys)
                if len(pressed) > 0:
                    if ec.button_handler.keys == []:  # this was first press
                        ec.button_handler.keys = pressed[0]  # only keep first
                        ec.button_handler.rt = \
                            ec.button_handler.clock.getTime()
                        continue_trial = False  # any response ends the trial

            # try to end trial, but check if we're really done
            if not continue_trial:
                break
            continue_trial = False
            for comp in ec.trial_components:
                if hasattr(comp, 'status') and comp.status != FINISHED:
                    continue_trial = True
                    break  # at least one trial component not finished

            # check for force-quit
            if event.getKeys(['escape']):
                core.quit()

            # screen flip
            if continue_trial:
                ec.win.flip()

        # trial is over
        if len(ec.button_handler.keys) == 0:
            ec.data_handler.addData('button_presses', None)
        else:
            ec.data_handler.addData('button_presses', ec.button_handler.keys)
            ec.data_handler.addData('reaction_times', ec.button_handler.rt)
        ec.data_handler.nextEntry()
        # some feedback
        if int(pressed[0]) == trial_order[n] + 1:
            running_total = running_total + 1
            ec.screen_prompt('Correct! Your reaction time was '
                             '{}'.format(np.round(ec.button_handler.rt, 3)))
        else:
            ec.screen_prompt('You pressed {0}, the correct answer was '
                             '{1}.'.format(pressed[0], trial_order[n] + 1))
        ec.wait_secs(feedback_dur)
        ec.clear_screen()
        ec.wait_secs(isi)

    # # # # # # # # # #
    # end experiment  #
    # # # # # # # # # #
    ec.screen_prompt('All done! You got {0} correct out of {1} '
                     'trials.'.format(running_total, num_trials))
    ec.wait_secs(feedback_dur)
    ec.clear_screen()
    core.quit()
