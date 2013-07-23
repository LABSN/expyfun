from expyfun import ExperimentController, set_log_level
import numpy as np

set_log_level('DEBUG')
with ExperimentController() as ec:
    # Make some audio data
    duration = 1  # in seconds
    t = np.arange(np.round(duration * ec.fs))
    audio = np.sin(2 * np.pi * 440 * t)
    # load the data
    ec.load_buffer(audio)
    # flip the screen and play the sound
    ec.flip_and_play()

print 'Done!'
