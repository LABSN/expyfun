.. -*- mode: rst -*-

.. image:: https://travis-ci.org/LABSN/expyfun.png
  :target: https://travis-ci.org/LABSN/expyfun/
.. image:: https://coveralls.io/repos/LABSN/expyfun/badge.png
  :target: https://coveralls.io/r/LABSN/expyfun
.. image:: https://zenodo.org/badge/doi/10.5281/zenodo.11541.png
  :target: http://dx.doi.org/10.5281/zenodo.11541
  
`expyfun`_
=========

This package is designed for audio-visual experiments with precise timing,
and includes functionality for Eyelink control. This package is designed
with the purpose that it be used by LABS^N at the University of Washington.
It is not designed for public use.

Therefore, while we welcome bug reports and suggestions from others,
NO SUPPORT IS GUARANTEED. Moreover, we can and will change the API as
necessary to suit the needs of the lab. Thus, use at your own risk.

Note that lab calibration logs can be stored here:

https://github.com/LABSN/expyfun/wiki/Calibration-log

Requirements:

- numpy/scipy
- matplotlib
- pyglet (a bleeding-edge ``tip.zip`` as of the end of May 2014), see the
  "development version" here:
    http://www.pyglet.org/download.html

Optional:

- pandas (some plotting functions)
- joblib (parallel processing)
- mne-python (CUDA filtering/resampling)
- pytables (HDF5 write/read)

System-level:
- git (for automated version downloading)


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
