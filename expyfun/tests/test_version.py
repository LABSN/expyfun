# -*- coding: utf-8 -*-
import os
from os import path as op
import warnings
import shutil

from nose.tools import assert_raises, assert_true, assert_equal

from expyfun import (ExperimentController, assert_version, download_version,
                     __version__)
from expyfun._utils import _TempDir
from expyfun._git import _has_git


def test_version():
    """Test version assertions."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        assert_raises(TypeError, assert_version, 1)
        assert_raises(TypeError, assert_version, '1' * 8)
        assert_raises(AssertionError, assert_version, 'x' * 7)
        assert_version(__version__[-7:])
    assert_true(all('actual' in str(ww.message) for ww in w))

    for want_version in ('cae6bc3', '090948e'):  # old, new
        tempdir = _TempDir()
        tempdir_2 = _TempDir()
        if not _has_git:
            assert_raises(ImportError, download_version, want_version, tempdir)
        else:
            assert_raises(IOError, download_version, want_version,
                          op.join(tempdir, 'foo'))
            assert_raises(RuntimeError, download_version, 'x' * 7, tempdir)
            download_version(want_version, tempdir)
            ex_dir = op.join(tempdir, 'expyfun')
            assert_true(op.isdir(ex_dir))
            assert_true(op.isfile(op.join(ex_dir, '__init__.py')))
            with open(op.join(ex_dir, '_version.py')) as fid:
                line1 = fid.readline().strip()
            if want_version == 'cae6bc3':
                assert_equal(line1.split(' = ')[1][-8:-1], '.dev0+c')
            else:
                assert_equal(line1.split(' = ')[1][-8:-1], want_version)

            # auto dir determination
            orig_dir = os.getcwd()
            os.chdir(tempdir)
            try:
                assert_true(op.isdir('expyfun'))
                assert_raises(IOError, download_version, want_version)
            finally:
                os.chdir(orig_dir)
            # make sure we can get latest version
            assert_raises(IOError, download_version, dest_dir=tempdir)
            download_version(dest_dir=tempdir_2)


def test_integrated_version_checking():
    """Test EC version checking during init."""
    tempdir = _TempDir()
    args = ['test']  # experiment name
    kwargs = dict(output_dir=tempdir, full_screen=False, window_size=(1, 1),
                  participant='foo', session='01', stim_db=0.0, noise_db=0.0,
                  verbose=True)
    assert_raises(RuntimeError, ExperimentController, *args, version=None,
                  **kwargs)
    assert_raises(AssertionError, ExperimentController, *args,
                  version='59f3f5b', **kwargs)  # the very first commit
