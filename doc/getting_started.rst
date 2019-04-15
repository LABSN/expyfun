Getting started
===============

.. contents::
   :local:
   :depth: 2

Installing expyfun
------------------

Python
^^^^^^
The first step is to install a Python distribution. See tutorials on other
sites for how to do this.

Dependencies
^^^^^^^^^^^^
expyfun requires several libraries for full functionality:


- Required Python libraries (can be installed via ``pip``, or some with ``conda``
  if preferred):

  - ``numpy``
  - ``scipy``
  - ``matplotlib``
  - ``pyglet``: 1.2.0 or later is recommended

- Optional Python libraries:

  - ``rtmixer``: for high precision audio playback, see :ref:`rtmixer_installation`.
  - ``TDTpy``: if using TDT on Windows
  - ``mne``: for filtering/resampling -- with CUDA if mne dependencies installed
  - ``pandas``: Some plotting functions
  - ``joblib``: Parallel processing
  - ``h5py``: HDF5 write/read

- Optional system software:

  - ``git``: command-line tools needed for automated version downloading
  - ``AVbin``: if playing compressed videos

Expyfun
^^^^^^^
The recommended way to install expyfun on
development machines is to ``git clone`` the reposity then do:

.. code-block:: console

    $ python setup.py develop
    $ pip install -e .

This allows you to stay up to date with updates, changes, and bugfixes,
and easily switch between versions.

Configuring expyfun
-------------------
expyfun is designed to "just run" on user machines regardless of OS (Windows,
macOS, or Linux) machines, and does not require additional configuration.
In this state, the A/V/trigger timing is not guaranteed, but should be
sufficient to work out most experiment logistics.

To configure expyfun on an experimental machine designed for precise A/V
timing, information TBD.

It is important to use :ref:`synchronization_tests` to validate your final
system delay and jitter properties.

Deploying experiments
---------------------
The function :func:`expyfun.download_version` should be used to deploy a
static version of expyfun once an experiment is in its finalized state.
