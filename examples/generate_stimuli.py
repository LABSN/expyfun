import numpy as np
from scipy.io import savemat, wavfile
from expyfun.experiment_controller import get_tdt_rates


def generate_stimuli(num_trials=10, num_freqs=4, stim_dur=0.5, min_freq=500.0,
                     max_freq=4000.0, fs=None, rms=0.01, save_as='mat'):
    """Make some sine waves and save in various formats. Optimized for saving
    as MAT files, but can also save directly as WAV files, or can return a
    python dictionary with sinewave data as values.

    Parameters
    ----------
    num_trials : int | 10
        Number of trials you want in your experiment. Ignored if save_as is
        not 'mat'.
    num_freqs : int | 4
        Number of frequencies (equally-spaced on a log2-scale) at which to
        generate tones.
    stim_dur : float | 0.5
        Duration of the tones in seconds.
    min_freq : float | 250
        Frequency of the lowest tone in Hertz.
    max_freq : float | 250
        Frequency of the highest tone in Hertz.
    fs : float | None
        Sampling frequency of resulting sinewaves.  Defaults to 24414.0625 (a
        standard rate for TDTs) if no value is specified.
    rms : float | 0.01
        RMS amplitude to which all sinwaves will be scaled
    save_as : str | 'mat'
        Format in which to return the sinewaves. 'dict' returns sinewave arrays
        as values in a python dictionary; 'wav' saves them as WAV files at
        sampling frequency 'fs'; 'mat' saves them as a MAT file along with
        related variables 'fs', 'freqs', 'trial_order', and 'rms'.
    """

    if fs is None:
        fs = get_tdt_rates()['25k']

    t = np.arange(np.round(stim_dur * fs)) / fs

    # frequencies equally spaced on a log-2 scale
    freqs = min_freq * np.logspace(0, np.log2(max_freq / min_freq), num_freqs,
                                   endpoint=True, base=2)

    # strings for the filenames / dictionary keys
    freq_names = map(str, [int(f) for f in freqs])
    index_names = map(str, range(num_freqs))
    names = ['f' + n + '_' + f for (n, f) in zip(index_names, freq_names)]

    # generate sinewaves & RMS normalize
    wavs = [np.sin(2 * np.pi * f * t) for f in freqs]
    wavs = [rms / np.sqrt(np.mean(w ** 2)) * w for w in wavs]

    # nasty hack, shouldn't be needed in future versions of scipy.io.wavfile:
    if save_as == 'wav':
        wavs = [np.int16(w / np.max(np.abs(w)) * 32767) for w in wavs]

    # collect into dictionary & save
    wav_dict = {n: w for (n, w) in zip(names, wavs)}
    if save_as == 'mat':
        num_reps = num_trials / num_freqs + 1
        trials = np.tile(range(num_freqs), num_reps)
        trial_order = np.random.permutation(trials[0:num_trials])
        wav_dict.update({'trial_order': trial_order, 'freqs': freqs, 'fs': fs,
                         'rms': rms})
        savemat('equally_spaced_sinewaves.mat', wav_dict, oned_as='row')
    elif save_as == 'wav':
        for n in names:
            wavfile.write(n + '.wav', int(fs), wav_dict[n])
    else:  # save_as 'dict'
        return wav_dict
