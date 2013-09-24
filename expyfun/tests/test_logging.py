import os

from expyfun._utils import _TempDir, tdt_test
from expyfun import ExperimentController

tempdir = _TempDir()
std_args = ['test']
std_kwargs = dict(participant='foo', session='01', full_screen=False,
                  window_size=(1, 1))


def test_logging(ac='psychopy'):
    """Test logging to file (PsychoPy)
    """
    os.chdir(tempdir)
    ec = ExperimentController(*std_args, audio_controller=ac, **std_kwargs)
    test_name = ec._log_file
    stamp = ec.current_time
    ec.wait_until(stamp)  # wait_until called with already passed timestamp
    try:
        ec.load_buffer([1., -1., 1., -1., 1., -1.])  # RMS warning
    except UserWarning:
        pass
    ec.close()
    with open(test_name) as fid:
        data = '\n'.join(fid.readlines())

    # check for various expected log messages (TODO: add more)
    should_have = ['Subject: foo', 'Session: 01', 'wait_until was called',
                   'Stimulus max RMS (']
    if ac == 'psychopy':
        should_have.append('PsychoPy')
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
