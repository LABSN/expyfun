# -*- coding: utf-8 -*-
import os
from os import path as op
from nose.tools import assert_raises, assert_true, assert_equal

from expyfun import (ExperimentController, assert_version, download_version,
                     __version__)
from expyfun._utils import _TempDir
from expyfun._git import _has_git

tempdir = _TempDir()
tempdir_2 = _TempDir()


def test_version():
    """Test version assertions
    """
    assert_raises(TypeError, assert_version, 1)
    assert_raises(TypeError, assert_version, '1' * 8)
    assert_raises(AssertionError, assert_version, 'x' * 7)
    assert_version(__version__[-7:])

    v = '7807b1b'
    f = 'drammock'
    if not _has_git:
        assert_raises(ImportError, download_version, v, tempdir, f)
    else:
        assert_raises(IOError, download_version, v, op.join(tempdir, 'foo'), f)
        assert_raises(RuntimeError, download_version, 'x' * 7, tempdir, f)
        download_version(v, tempdir, f)
        ex_dir = op.join(tempdir, 'expyfun')
        assert_true(op.isdir(ex_dir))
        assert_true(op.isfile(op.join(ex_dir, '__init__.py')))
        with open(op.join(ex_dir, '_version.py')) as fid:
            line1 = fid.readline().strip()
            line2 = fid.readline().strip()
        assert_equal(line1.split(' = ')[1][-8:-1], v)
        assert_equal(line2.split(' = ')[1].strip('\''), f)

        # auto dir determination
        orig_dir = os.getcwd()
        os.chdir(tempdir)
        try:
            assert_true(op.isdir('expyfun'))
            assert_raises(IOError, download_version, v)
        finally:
            os.chdir(orig_dir)
        # make sure we can get latest version
        download_version(dest_dir=tempdir_2)


def test_integrated_version_checking():
    """Test EC version checking during init
    """
    args = ['test']  # experiment name
    kwargs = dict(output_dir=tempdir, full_screen=False, window_size=(1, 1),
                  participant='foo', session='01', stim_db=0.0, noise_db=0.0,
                  verbose=True)
    assert_raises(RuntimeError, ExperimentController, *args, version=None,
                  **kwargs)
    assert_raises(AssertionError, ExperimentController, *args,
                  version='59f3f5b', **kwargs)  # the very first commit
