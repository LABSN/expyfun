#! /usr/bin/env python
#
# Copyright (C) 2013 Dan McCloy <drmccloy@uw.edu>
#    Re-used based on mne-python code.

import os
import expyfun

import setuptools  # we are using a setuptools namespace
from numpy.distutils.core import setup

descr = """Experiment controller functions."""

DISTNAME            = 'expyfun'
DESCRIPTION         = descr
MAINTAINER          = 'Dan McCloy'
MAINTAINER_EMAIL    = 'drmccloy@uw.edu'
URL                 = 'http://github.com/LABSN/expyfun'
LICENSE             = 'BSD (3-clause)'
DOWNLOAD_URL        = 'http://github.com/LABSN/expyfun'
VERSION             = expyfun.__version__


if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          include_package_data=True,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=open('README.rst').read(),
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
          packages=['expyfun', 'expyfun.tests'],
          package_data={},
          scripts=[])


