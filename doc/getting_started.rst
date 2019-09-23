:orphan:

Getting started
===============

.. contents::
   :local:
   :depth: 2

Installing expyfun
------------------

.. highlight:: console

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
  - ``pyglet``: 1.2.0 or later is recommended but 1.4.0 is not fully supported
    yet, so use e.g. ``pip install "pyglet<1.4"``.

- Optional libraries:

  - ``rtmixer``: High precision audio playback, can be installed with
    ``pip install rtmixer``. On Linux you'll also need the ``libportaudio2``
    system package.
  - ``pyparallel`` or ``inpout32.dll``: Parallel port triggering,
    see :ref:`parallel_installation`.
  - ``TDTpy``: Using the TDT (only available on Windows).
  - ``mne``:  Filtering and resampling stimuli.
  - ``pandas``: Required for some plotting functions.
  - ``joblib``: Parallel processing
  - ``h5py``: HDF5-based writing and reading.

- Optional system software:

  - ``git``: Command-line tools needed for automated version downloading.
  - ``FFmpeg`` or ``AVBin`` (Pyglet >= or < 1.4, respectively): For playing compressed videos.

Expyfun
^^^^^^^
The recommended way to install expyfun on
development machines is to ``git clone`` the reposity then do:

    $ pip install -e .

This allows you to stay up to date with updates, changes, and bugfixes,
and easily switch between versions.

Configuring expyfun
-------------------
expyfun is designed to "just run" on user machines regardless of OS (Windows,
macOS, or Linux) machines, and does not require additional configuration.
In this state, the A/V/trigger timing is not guaranteed, but should be
sufficient to work out most experiment logistics.

To configure expyfun on an experimental machine designed for precise
A/V/trigger timing typically requires utilizing:

- oscilloscope
- photodiode
- parallel port breakout or TDT trigger breakout
- auditory connectors to go 1/4" or 1/8" output->BNC
- Running :ref:`synchronization_tests`

Once the jitter is sufficiently reduced (e.g., < 1 ms) and delays between
components are fixed, delays can be corrected through various config
variable(s) (details TBD).

Deploying experiments
---------------------
The function :func:`expyfun.download_version` should be used to deploy a
static version of expyfun once an experiment is in its finalized state.
