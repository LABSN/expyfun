r"""
======================
Audio sample rate test
======================

This example tests the TDT sample rate, which we expect to be ``24414.0625``.
To test this:

1. Connect the TDT audio output to the sound card input. This can be on
   the same machine that the TDT is connected to, or a different one.
   A 1/8" male-male cable running from the headphone monitor to the
2. Start Audacity on the sound-card machine.
3. Configure the sound-card machine and/or Audacity to record from the sound
   card input.
4. Tell Audacity to record.
5. Run this script. It should take about 40 seconds.
6. When the script completes, stop the Audacity recording.
7. Visually inspect the audacity recording for the time of the two sinc peaks.
   One peak should occur toward the beginning and the other toward the end.
8. The sound that was played put 1e6 (1,000,000) samples between the two
   peaks. So you can get the effective sample rate as:

   .. math::

       f_s = \frac{1000000}{t_{stop} - t_{start}}

   For example, Eric's RM1 (2017/01/10) had a start time of ``19.854330`` sec
   and an end time of ``60.813690`` sec for a difference of of ``40.95936``,
   yielding an effective sample rate of ``24414.44`` Hz.

If the audio output can also be connected simultaneously to other equipment,
e.g., an EEG system, the output can be split (or left/right outputs used) to
test multiple system synchronization at once.
"""
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

from expyfun import ExperimentController, building_doc

print(__doc__)

stim = np.zeros(int(1e6) + 1)
stim[[0, -1]] = 1.
stim_dur = len(stim) / 24414.
with ExperimentController('FsTest', full_screen=False, noise_db=-np.inf,
                          participant='s', session='0', output_dir=None,
                          suppress_resamp=True, check_rms=None,
                          version='dev') as ec:
    ec.identify_trial(ec_id='', ttl_id=[0])
    ec.load_buffer(stim)
    print('Starting stimulus')
    ec.start_stimulus()
    print('Stimulus started')
    if not building_doc:
        ec.wait_secs(stim_dur + 1.)
    ec.stop()
    print('Stimulus done')
