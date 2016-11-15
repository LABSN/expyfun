import os
import warnings

from expyfun._utils import _TempDir, _hide_window
from expyfun import ExperimentController

warnings.simplefilter('always')

std_args = ['test']
std_kwargs = dict(participant='foo', session='01', full_screen=False,
                  window_size=(1, 1), verbose=True, noise_db=0, version='dev')


@_hide_window
def test_logging(ac='pyglet'):
    """Test logging to file (Pyglet)
    """
    tempdir = _TempDir()
    orig_dir = os.getcwd()
    os.chdir(tempdir)
    try:
        with ExperimentController(*std_args, audio_controller=ac,
                                  response_device='keyboard',
                                  trigger_controller='dummy',
                                  **std_kwargs) as ec:
            test_name = ec._log_file
            stamp = ec.current_time
            ec.wait_until(stamp)  # wait_until called w/already passed timest.
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                ec.load_buffer([1., -1., 1., -1., 1., -1.])  # RMS warning

        with open(test_name) as fid:
            data = '\n'.join(fid.readlines())

        # check for various expected log messages (TODO: add more)
        should_have = ['Subject: foo', 'Session: 01', 'wait_until was called',
                       'Stimulus max RMS (']
        if ac == 'pyglet':
            should_have.append('Pyglet')
        else:
            should_have.append('TDT')

        for s in should_have:
            if s not in data:
                raise ValueError('Missing data: "{0}" in:\n{1}'
                                 ''.format(s, data))
    finally:
        os.chdir(orig_dir)


def test_logging_tdt():
    """Test logging to file (TDT)
    """
    test_logging('tdt')
