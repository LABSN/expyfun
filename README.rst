.. -*- mode: rst -*-

.. image:: https://travis-ci.org/LABSN/expyfun.png
  :target: https://travis-ci.org/LABSN/expyfun/
.. image:: https://ci.appveyor.com/api/projects/status/yvep4fd9tv3t45r4/branch/master
  :target: https://ci.appveyor.com/project/Eric89GXL/expyfun/branch/master
.. image:: https://coveralls.io/repos/LABSN/expyfun/badge.png
  :target: https://coveralls.io/r/LABSN/expyfun
.. image:: https://zenodo.org/badge/doi/10.5281/zenodo.11640.png
  :target: http://dx.doi.org/10.5281/zenodo.11640
.. image:: https://badges.gitter.im/LABSN/expyfun.svg
  :alt: Join the chat at https://gitter.im/LABSN/expyfun
  :target: https://gitter.im/LABSN/expyfun?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

`expyfun`_
=========

This package is designed for audio-visual experiments with precise timing,
and includes functionality for Eyelink control. This package is designed
with the purpose that it be used by LABS^N at the University of Washington.
It is not designed for public use.

Therefore, while we welcome bug reports and suggestions from others,
NO SUPPORT IS GUARANTEED. Moreover, we can and will change the API as
necessary to suit the needs of the lab. Thus, use at your own risk.

Note that lab calibration logs can be stored `here
<https://github.com/LABSN/expyfun/wiki/Calibration-log>`_.

`The API documentation and examples can be found here
<https://labsn.github.io/expyfun>`_.

Requirements:

- numpy/scipy/matplotlib
- pyglet 1.2.0 or later
- TDTpy (if using TDT on Windows)
- mne-python (filtering/resampling -- with CUDA if mne dependencies installed)

Optional:

- pandas (some plotting functions)
- joblib (parallel processing)
- h5py (HDF5 write/read)

System-level:
- git (for automated version downloading)
- AVbin (if playing compressed videos)


Licensing
^^^^^^^^^

expyfun is **BSD-licenced** (3 clause):

    This software is OSI Certified Open Source Software.
    OSI Certified is a certification mark of the Open Source Initiative.

    Copyright (c) 2013, authors of expyfun
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the names of expyfun authors nor the names of any
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    **This software is provided by the copyright holders and contributors
    "as is" and any express or implied warranties, including, but not
    limited to, the implied warranties of merchantability and fitness for
    a particular purpose are disclaimed. In no event shall the copyright
    owner or contributors be liable for any direct, indirect, incidental,
    special, exemplary, or consequential damages (including, but not
    limited to, procurement of substitute goods or services; loss of use,
    data, or profits; or business interruption) however caused and on any
    theory of liability, whether in contract, strict liability, or tort
    (including negligence or otherwise) arising in any way out of the use
    of this software, even if advised of the possibility of such
    damage.**


.. image:: https://badges.gitter.im/LABSN/expyfun.svg
   :alt: Join the chat at https://gitter.im/LABSN/expyfun
   :target: https://gitter.im/LABSN/expyfun?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge