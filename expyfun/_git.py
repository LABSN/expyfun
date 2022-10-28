# -*- coding: utf-8 -*-
import os
from os import path as op
import sys
import warnings

from ._utils import _TempDir, string_types, run_subprocess, StringIO, reload
from ._version import __version__

this_version = __version__[-7:]

try:
    run_subprocess(['git', '--help'])
except Exception as exp:
    _has_git, why_not = False, str(exp)
else:
    _has_git, why_not = True, None


def _check_git():
    """Helper to check the expyfun version"""
    if not _has_git:
        raise RuntimeError('git not found: {0}'.format(why_not))


def _check_version_format(version):
    """Helper to ensure version is of proper format"""
    if not isinstance(version, string_types) or len(version) != 7:
        raise TypeError('version must be a string of length 7, got {0}'
                        ''.format(version))


def _active_version(wd):
    """Helper to get the currently active version"""
    return run_subprocess(['git', 'rev-parse', 'HEAD'], cwd=wd)[0][:7]


def download_version(version='current', dest_dir=None):
    """Download specific expyfun version

    Parameters
    ----------
    version : str
        Version to check out (7-character git commit number).
        Can also be ``'current'`` (default) to download whatever the
        latest ``upstream/master`` version is.
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
    repo_url = 'https://github.com/LABSN/expyfun.git'
    run_subprocess(['git', 'clone', repo_url, expyfun_dir])
    version = _active_version(expyfun_dir) if version == 'current' else version
    try:
        run_subprocess(['git', 'checkout', version], cwd=expyfun_dir)
    except Exception as exp:
        raise RuntimeError('Could not check out version {0}: {1}'
                           ''.format(version, str(exp)))
    assert _active_version(expyfun_dir) == version

    # install
    orig_dir = os.getcwd()
    os.chdir(expyfun_dir)
    # ensure our version-specific "setup" is imported
    sys.path.insert(0, expyfun_dir)
    orig_stdout = sys.stdout
    try:
        # on pytest with Py3k this can be problematic
        if 'setup' in sys.modules:
            del sys.modules['setup']
        import setup
        reload(setup)
        setup_version = setup.git_version()
        # This is necessary because for a while git_version returned
        # a tuple of (version, fork)
        if not isinstance(setup_version, string_types):
            setup_version = setup_version[0]
        assert version.lower() == setup_version[:7].lower()
        del setup_version
        sys.stdout = StringIO()
        with warnings.catch_warnings(record=True):  # PEP440
            setup.setup_package(
                script_args=['build', '--build-purelib', dest_dir])
    finally:
        sys.stdout = orig_stdout
        sys.path.pop(sys.path.index(expyfun_dir))
        os.chdir(orig_dir)
    print('\n'.join(['Successfully checked out expyfun version:', version,
                     'into destination directory:', op.join(dest_dir)]))


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
