import os
import warnings

from expyfun._utils import _TempDir, tdt_test
from expyfun import ExperimentController

warnings.simplefilter('always')

tempdir = _TempDir()
std_args = ['test']
std_kwargs = dict(participant='foo', session='01', full_screen=False,
                  window_size=(1, 1), verbose=True)


def test_logging(ac='pyo'):
    """Test logging to file (Pyo)
    """
    os.chdir(tempdir)
    with ExperimentController(*std_args, audio_controller=ac,
                              **std_kwargs) as ec:
        test_name = ec._log_file
        stamp = ec.current_time
        ec.wait_until(stamp)  # wait_until called with already passed timestamp
        with warnings.catch_warnings(True):
            ec.load_buffer([1., -1., 1., -1., 1., -1.])  # RMS warning

    with open(test_name) as fid:
        data = '\n'.join(fid.readlines())

    # check for various expected log messages (TODO: add more)
    should_have = ['Subject: foo', 'Session: 01', 'wait_until was called',
                   'Stimulus max RMS (']
    if ac == 'pyo':
        should_have.append('Pyo')
    else:
        should_have.append('TDT')

    for s in should_have:
        if not s in data:
            raise ValueError('Missing data: "{0}" in:\n{1}'.format(s, data))


@tdt_test
def test_logging_tdt():
    """Test logging to file (TDT)
    """
    test_logging('tdt')
