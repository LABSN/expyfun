# -*- coding: utf-8 -*-
import os
from os import path as op
from nose.tools import assert_raises, assert_true, assert_equal

from expyfun import assert_version, download_version, __version__
from expyfun._utils import _TempDir
from expyfun._git import _has_git

tempdir = _TempDir()


def test_version():
    """Test version assertions
    """
    assert_raises(TypeError, assert_version, 1)
    assert_raises(TypeError, assert_version, '1' * 8)
    assert_raises(AssertionError, assert_version, 'x' * 7)
    assert_version(__version__[-7:])

    v = '36e7da0'  # XXX CHANGE TO LAST VERSION AFTER MERGE!
    if not _has_git:
        assert_raises(ImportError, download_version, v, tempdir)
    else:
        assert_raises(IOError, download_version, v, op.join(tempdir, 'foo'))
        assert_raises(RuntimeError, download_version, 'x' * 7, tempdir)
        download_version(v, tempdir)
        ex_dir = op.join(tempdir, 'expyfun')
        assert_true(op.isdir(ex_dir))
        assert_true(op.isfile(op.join(ex_dir, '__init__.py')))
        with open(op.join(ex_dir, '_version.py')) as fid:
            line = fid.readline().strip()
        assert_equal(line.split(' = ')[1][-8:-1], v)

        # auto dir determination
        orig_dir = os.getcwd()
        os.chdir(tempdir)
        try:
            assert_true(op.isdir('expyfun'))
            assert_raises(IOError, download_version, v)
        finally:
            os.chdir(orig_dir)
