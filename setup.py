#! /usr/bin/env python
#
# Copyright (C) 2013 Dan McCloy <drmccloy@uw.edu>
#    Re-used based on mne-python code.

import os
import subprocess

# we are using a setuptools namespace
import setuptools  # analysis:ignore
from numpy.distutils.core import setup

descr = """Experiment controller functions."""

DISTNAME = 'expyfun'
DESCRIPTION = descr
MAINTAINER = 'Dan McCloy'
MAINTAINER_EMAIL = 'drmccloy@uw.edu'
URL = 'http://github.com/LABSN/expyfun'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'http://github.com/LABSN/expyfun'

# version
version_file = os.path.join('expyfun', '_version.py')
with open(version_file, 'r') as fid:
    line = fid.readline()
    VERSION = line.strip().split(' = ')[1][1:-1]


def git_version():
    """Helper adapted from Numpy"""
    def _minimal_ext_cmd(cmd):
        # minimal env; LANGUAGE is used on win32
        env = dict(LANGUAGE='C', LANG='C', LC_ALL='C')
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        return subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                env=env).communicate()[0]
    if os.path.exists('.git'):
        try:
            out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
            GIT_REVISION = out.decode('utf-8').strip()
        except OSError:
            GIT_REVISION = "Unknown"
        try:
            cmd = ['git', 'config', '--get', 'remote.origin.url']
            out = _minimal_ext_cmd(cmd).decode('utf-8').strip()
            start_idx = 0
            for prefix in ['git@github.com:', 'git://github.com/',
                           'http://github.com/', 'https://github.com/',
                           'ssh://git@github.com/']:
                if out.startswith(prefix):
                    start_idx = len(prefix)
            if start_idx:
                FORK = out[start_idx:out.rindex('/')]
        except OSError:
            FORK = 'Unknown'
    else:
        GIT_REVISION = "Unknown"
        FORK = 'Unknown'
    return [GIT_REVISION[:7], FORK]

FULL_VERSION = VERSION + '+' + git_version()[0]
FORK = git_version()[1]


def write_version(version, fork='Unknown'):
    with open(version_file, 'w') as fid:
        fid.write('__version__ = \'{0}\'\n'.format(version))
        fid.write('__fork__ = \'{0}\'\n'.format(fork))


def setup_package(script_args=None):
    """Actually invoke the setup call"""
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')
    with open('README.rst') as fid:
        long_description = fid.read()
    kwargs = dict(
        name=DISTNAME,
        maintainer=MAINTAINER,
        include_package_data=True,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        version=FULL_VERSION,
        download_url=DOWNLOAD_URL,
        long_description=long_description,
        zip_safe=False,  # the package can run out of an .egg file
        classifiers=['Intended Audience :: Science/Research',
                     'Intended Audience :: Developers',
                     'License :: OSI Approved',
                     'Programming Language :: Python',
                     'Topic :: Software Development',
                     'Topic :: Scientific/Engineering',
                     'Operating System :: Microsoft :: Windows',
                     'Operating System :: POSIX',
                     'Operating System :: Unix',
                     'Operating System :: MacOS'],
        platforms='any',
        packages=['expyfun', 'expyfun.tests',
                  'expyfun.analyze', 'expyfun.analyze.tests',
                  'expyfun.codeblocks',
                  'expyfun._externals',
                  'expyfun.io', 'expyfun.io.tests',
                  'expyfun.stimuli', 'expyfun.stimuli.tests',
                  'expyfun.visual', 'expyfun.visual.tests'],
        package_data={'expyfun': [os.path.join('data', '*')]},
        scripts=[])
    if script_args is not None:
        kwargs['script_args'] = script_args
    try:
        write_version(FULL_VERSION, FORK)
        setup(**kwargs)
    finally:
        write_version(VERSION, FORK)


if __name__ == "__main__":
    setup_package()
