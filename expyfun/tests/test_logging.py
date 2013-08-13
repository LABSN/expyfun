import os

from expyfun.utils import _TempDir
from expyfun import ExperimentController

tempdir = _TempDir()
std_args = ['test']
std_kwargs = dict(participant='foo', session='01', full_screen=False,
                  window_size=(1, 1))


def test_logging():  # rename once fixed
    """Test logging (to file)
    """
    os.chdir(tempdir)
    ec = ExperimentController(*std_args, **std_kwargs)
    test_name = ec._log_file
    ec.close()
    with open(test_name) as fid:
        data = '\n'.join(fid.readlines())
    # This list should be better
    should_have = ['PsychoPy', 'Subject: foo', 'Session: 01']  # XXX Add more
    for s in should_have:
        if not s in data:
            raise ValueError('Missing data: "{0}" in:\n{1}'
                             ''.format(s, data))
