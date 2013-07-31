import numpy as np
from scipy.io import savemat, wavfile
from expyfun.experiment_controller import get_tdt_rates


def generate_stimuli(num_freq=2, stim_dur=0.5, min_freq=500.0,
                     max_freq=4000.0, save_as='dict', fs=None):
    """Make some sine waves and save in various formats.

    Parameters
    ----------
    num_freq : int | 2
        Number of equally-spaced freqencies at which to generate tones.
    stim_dur : float | 0.5
        Duration of the tones in seconds.
    min_freq : float | 250
        Frequency of the lowest tone in Hertz.
    max_freq : float | 250
        Frequency of the highest tone in Hertz.
    save_as : str | 'dict'
        Format in which to return the sinewaves. 'dict' returns sinewave arrays
        as values in a python dictionary; 'mat' saves them as a MAT file; 'wav'
        saves them as WAV files at a sampling frequency 'fs'.
    fs : float | None
        Sampling frequency of resulting sinewaves.  Defaults to 24414.0625 (a
        standard rate for TDTs) if no value is specified.
    """

    if fs is None:
        fs = get_tdt_rates()['25k']

    t = np.arange(np.round(stim_dur * fs)) / fs
    freqs = min_freq * np.logspace(0, np.log2(max_freq / min_freq), num_freq,
                                   endpoint=True, base=2)

    freq_names = map(str, [int(f) for f in freqs])
    index_names = map(str, range(num_freq))
    names = ['f' + n + '_' + f for (n, f) in zip(index_names, freq_names)]
    wavs = [np.sin(2 * np.pi * f * t) for f in freqs]

    # nasty hack, shouldn't be needed in future versions of scipy.io.wavfile:
    if save_as == 'wav':
        wavs = [np.int16(w / np.max(np.abs(w)) * 32767) for w in wavs]

    wav_dict = {n: w for (n, w) in zip(names, wavs)}
    if save_as == 'dict':
        return wav_dict
    if save_as == 'wav':
        for n in names:
            wavfile.write(n + '.wav', fs, wav_dict[n])
    else:
        savemat('equally_spaced_sinewaves.mat', wav_dict, oned_as='row')