from expyfun import ExperimentController, set_log_level
import numpy as np
from scipy import io as sio

set_log_level('DEBUG')

noise_amp = 45  # dB for background noise
stim_amp = 75  # dB for stimuli
min_resp_time = 0.1
max_resp_time = 1.0
feedback_dur = 2.0

# generate some stimuli
# generate_stimuli(4, save_as='mat')

# load some stimuli (see generate_stimuli.py)
stims = sio.loadmat('equally_spaced_sinewaves.mat')
orig_rms = stims['rms']
freqs = stims['freqs']
fs = stims['fs']
trial_order = stims['trial_order']
num_trials = len(trial_order)
num_freqs = len(freqs)

if num_freqs > 8:
    raise ExperimentController.ExperimentError('Too many frequencies / '
                                               'not enough buttons.')

# keep only the sinusoid data
wavs = {k: stims[k] for k in stims if k not in ('rms', 'fs', 'freqs',
                                                'trial_order')}
# response, correct?, time
responses = np.zeros(num_trials, 3)
# screen flip, audio start, and when responses stopped being listened for
timestamps = np.zeros(num_trials, 3)

# instructions
instructions = ('You will hear tones at {0} different frequencies. Your job is to'
                'press the button corresponding to that frequency. Please press'
                'each button to hear its corresponding tone.'.format(len(freqs)))

with ExperimentController('testExp', 'RM1', 'keyboard', stim_ampl=75,
                          noise_ampl=45) as ec:
    with open(ec.exp_info['exp_name'] + '_output.tab', 'w') as f:
        f.write('some stuff')  # TODO: output headers
        # TODO: define flip and play function to take timestamps
        #ec.call_on_flip_and_play()
        for t in range(num_trials):
            # load the data
            ec.load_buffer(wavs[wavs.keys()[t]])
            # flip the screen and play the sound
            ec.flip_and_play()
            # TODO: wait for button press, get data
            f.write('some data')  # TODO: save data
print 'Done!'





""" MOVE THIS TO generate_stimuli.py

    % Before starting trials, let's give some instructions
    instructions = sprintf(['Welcome to the experiment!\n\nYou will hear ' ...
        'tones at %i different frequencies. Your job is to press the button ' ...
        'corresponding to that frequency.\n\nPlease press a button to hear ' ...
        'each tone and see its corresponding button number. (Hint: they go ' ...
        'in ascending order!)'],nFreqs);
    ScreenPrompt(instructions, display, TDT);
    WaitSecs(0.5);

    % Now let's play each tone with the corresponding button number,
    % separated temporally by one second
    showNumberDur = 0.4;
    interToneInterval = 1.0;
    tPrevious = 0;
    for fi = 1:nFreqs
        % It is ABSOLUTELY CRITICAL to scale the stimuli correctly. Hearing
        % damage can occur if this is done incorrectly!
    	waveData = stimScaler * wavs{fi};
        AudioController('loadBuffer', TDT, waveData);
        DrawFormattedText(display.windowPtr, num2str(fi), 'center', 'center', display.scrWhite, 80, 0, 0, 2 );
        tFlip = Screen('Flip', display.windowPtr, tPrevious + interToneInterval);
        tPrevious = tFlip;
        AudioController('start', TDT)
        tFlip = Screen('Flip', display.windowPtr, tFlip + showNumberDur);
        % Wait until auditory stimulus is done playing
        WaitSecs(tFlip + length(waveData) / fs - GetSecs());
        AudioController('stopReset', TDT);
    end

    instructions = sprintf(['Now you''re ready for the experiment. ' ...
        'Press the response button corresponding to each tone as quickly ' ...
        'as possible after it is played. Press 1 or 2 to begin.']);
    ScreenPrompt(instructions, display, TDT);
    WaitSecs(1);

    for trialNum = 1:nTrials
        % Let's draw a fixation dot that spans 1 degree of visual angle
        fixWidth = round(ceil(deg2pix(display,1)));
        fixBox = vector(display.center.'*[1 1] + [-1 1; -1 1] * fixWidth/2);
        Screen('FillOval', display.windowPtr, display.scrWhite, fixBox);
        Screen('Flip', display.windowPtr);

        % Make sure we don't go too quickly from trial to trial
        WaitSecs(1.0);
        % Figure out the correct stimulus to run
		ii = trialOrder(trialNum);

		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%  Load sound stimuli into TDT  %
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		waveData = stimScaler * wavs{ii};
		% Ensure any previous stimuli are cleared before loading new one
		AudioController('clearBuffer', TDT);
		AudioController('loadBuffer', TDT, waveData);

		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%  Cue Frame and audio playback start  %
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   	    Screen('FillOval', display.windowPtr, display.scrWhite, fixBox);
		tTrialStart = Screen('Flip', display.windowPtr);
		tSound = AudioController('start', TDT);

		%%%%%%%%%%%%%%%
		%  Responses  %
		%%%%%%%%%%%%%%%
        [num, pressTime] = waitForButtonPress(TDT,1:nFreqs,minRespTime,maxRespTime);
        respList(trialNum,:) = [num, pressTime];
		AudioController('stopReset', TDT);

		%%%%%%%%%%%%%
		%  Feedback %
		%%%%%%%%%%%%%
        WaitSecs(0.5);
        correctness(trialNum) = ii == num;
        if correctness(trialNum)
            feedback = 'Correct!';
        elseif ~isnan(num)
            feedback = sprintf(['Incorrect -- you pressed %i and the '...
                'correct response was %i.'], num, ii);
        else
            feedback = 'Response was too slow, try to be faster!';
        end
        ScreenPrompt(feedback,display,TDT,1:nFreqs,feedbackDur);
		tDone = GetSecs();

        % Save the data in raw form after every trial!
		timeVecs(:,trialNum) = [tTrialStart tSound tDone];
		temp = datestr(clock);
		temp(temp==' ' | temp==':')='_';
		timeStopped = temp;
		save(saveFile,'respList','trialOrder','freqs','trialNum','timeStopped','timeVecs','correctness');
		quitCheck;
    end
    fprintf('Performance was %0.2f%%.\n',100*mean(correctness));
	cleanupError(TDT);
catch err
	cleanupError(TDT,err);
end
