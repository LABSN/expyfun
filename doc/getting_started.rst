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
  - ``pyglet``

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

.. code-block:: console

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

To get this to work, you'll need to set up the machine configuration file. This
ensures that the following things (among others) work correctly:

1. The interface for auditory stimuli.
2. The interface for triggering.
3. Units, e.g., ``'deg'`` actually yields degrees.
4. The display screen resolution in full-screen mode.

The keys that will always need to be set (using :func:`expyfun.set_config` or
manual JSON editing) include, but are not limited to (all *distances* in cm;
example values from a fairly typical desktop computer):

``"SCREEN_SIZE_PIX"``
    Full screen size in pixels.
``"SCREEN_DISTANCE"``
    Physical display distance from the subject, e.g., ``"83.0"``.
``"SCREEN_WIDTH"``
    Physical display width, e.g., ``"52.0"``.

.. note::

     Another settable parameter is ``"SCREEN_HEIGHT"``, but if you have square
     display pixels (a sane assumption for reasonable displays) then it's
     inferred based on the screen size in pixels and physical screen width.

Other settings depend on whether you use TDT / sound card / parallel port for
auditory stimuli and triggering. Possibilities can be seen by looking at
:py:obj:`expyfun.known_config_types`. Your current system configuration can be
viewed by doing::

    >>> expyfun.get_config()
    {'SCREEN_DISTANCE': '61.0', 'SCREEN_SIZE_PIX': '1920,1200', 'SCREEN_WIDTH': '52.0', 'SOUND_CARD_BACKEND': 'rtmixer'}

Deploying experiments
---------------------
The function :func:`expyfun.download_version` should be used to deploy a
static version of expyfun once an experiment is in its finalized state.
