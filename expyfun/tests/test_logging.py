from numpy.testing import assert_equal
from nose.tools import assert_true, assert_raises
import os.path as op
import os
import warnings

from expyfun.utils import (_TempDir, get_config, set_config, deprecated,
                           set_log_file, set_log_level)
from expyfun import ExperimentController

base_dir = op.join(op.dirname(__file__), 'data')
fname_log = op.join(base_dir, 'test-ec.log')
tempdir = _TempDir()
test_name = op.join(tempdir, 'test.log')


def compare_logs(fname_1, fname_2):
    """Helper to read, parse, and compare logs
    """


def old_logging():  # rename once fixed
    """Test logging (to file)
    """
    # Made a correct output example, saved it to test-ec.log to compare
    with open(fname_log, 'r') as old_log_file:
        old_lines = old_log_file.readlines()

    if op.isfile(test_name):
        os.remove(test_name)

    # XXX test it with printing default off
    set_log_file(test_name)
    set_log_level('WARNING')
    # should NOT print
    ec = ExperimentController()
    assert_true(open(test_name).readlines() == [])
    # should NOT print
    ec = ExperimentController(verbose=False)
    assert_true(open(test_name).readlines() == [])
    # should NOT print
    ec = ExperimentController(verbose='WARNING')
    assert_true(open(test_name).readlines() == [])
    # SHOULD print
    ec = ExperimentController(verbose=True)
    new_log_file = open(test_name, 'r')
    new_lines = clean_lines(new_log_file.readlines())
    assert_equal(new_lines, old_lines)
    new_log_file.close()
    set_log_file(None)  # Need to do this to close the old file
    os.remove(test_name)

    # now go the other way (printing default on)
    set_log_file(test_name)
    set_log_level('INFO')
    # should NOT print
    ec = ExperimentController(verbose='WARNING')
    assert_true(open(test_name).readlines() == [])
    # should NOT print
    ec = ExperimentController(verbose=False)
    assert_true(open(test_name).readlines() == [])
    # SHOULD print
    ec = ExperimentController()
    new_log_file = open(test_name, 'r')
    old_log_file = open(fname_log, 'r')
    new_lines = clean_lines(new_log_file.readlines())
    assert_equal(new_lines, old_lines)
    # check to make sure appending works (and as default, raises a warning)
    with warnings.catch_warnings(True) as w:
        set_log_file(test_name, overwrite=False)
        assert len(w) == 0
        set_log_file(test_name)
        assert len(w) == 1

    # make sure overwriting works
    set_log_file(test_name, overwrite=True)
    # this line needs to be called to actually do some logging
    ec = ExperimentController()
    del ec
    new_log_file = open(test_name, 'r')
    new_lines = clean_lines(new_log_file.readlines())
    assert_equal(new_lines, old_lines)
