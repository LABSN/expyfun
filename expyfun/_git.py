# -*- coding: utf-8 -*-
import os
from os import path as op
import sys

from ._utils import _TempDir, string_types, run_subprocess
from ._version import __version__

this_version = __version__[-7:]


try:
    run_subprocess(['git', '--help'])
except Exception as exp:
    _has_git, why_not, git = False, str(exp), None
else:
    _has_git, why_not = True, None


def _check_git():
    """Helper to check the expyfun version"""
    if not _has_git:
        raise RuntimeError('git not found: {0}'.format(str(exp)))


def _check_version_format(version):
    """Helper to ensure version is of proper format"""
    if not isinstance(version, string_types) or len(version) != 7:
        raise TypeError('version must be a string of length 7, got {0}'
                        ''.format(version))


def download_version(version, dest_dir=None):
    """Download specific expyfun version

    Parameters
    ----------
    version : str
        Version to check out (7-character git commit number).
    dest_dir : str | None
        Destination directory. If None, the current working
        directory is used.

    Notes
    -----
    This function requires installation of ``gitpython``.
    """
    _check_git()
    _check_version_format(version)
    if dest_dir is None:
        dest_dir = os.getcwd()
    if not isinstance(dest_dir, string_types) or not op.isdir(dest_dir):
        raise IOError('Destination directory {0} does not exist'
                      ''.format(dest_dir))
    if op.isdir(op.join(dest_dir, 'expyfun')):
        raise IOError('Destination directory {0} already has "expyfun" '
                      'subdirectory'.format(dest_dir))

    # fetch locally and get the proper version
    tempdir = _TempDir()
    expyfun_dir = op.join(tempdir, 'expyfun')  # git will auto-create this dir
    repo_url = 'git://github.com/LABSN/expyfun.git'
    run_subprocess(['git', 'clone', repo_url, expyfun_dir])
    try:
        run_subprocess(['git', 'checkout', version], cwd=expyfun_dir)
    except Exception as exp:
        raise RuntimeError('Could not check out version {0}: {1}'
                           ''.format(version, str(exp)))

    # install
    orig_dir = os.getcwd()
    os.chdir(expyfun_dir)
    sys.path.insert(0, expyfun_dir)  # ensure our new "setup" is imported
    try:
        from setup import git_version, setup_package
        assert git_version().lower() == version[:7].lower()
        setup_package(script_args=['build', '--build-purelib', dest_dir])
    finally:
        sys.path.pop(sys.path.index(expyfun_dir))
        os.chdir(orig_dir)


def assert_version(version):
    """Assert that a specific version of expyfun is imported

    Parameters
    ----------
    version : str
        Version to check (7 characters).
    """
    _check_version_format(version)
    if this_version.lower() != version.lower():
        raise AssertionError('Requested version {0} does not match current '
                             'version {1}'.format(version, this_version))
