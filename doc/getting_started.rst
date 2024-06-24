Getting started
===============

.. contents::
   :local:
   :depth: 2

Installing expyfun
------------------

.. highlight:: python

Python
^^^^^^
The first step is to install a Python 3.8+ distribution. See tutorials on other
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
  - ``pillow``

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
  - ``FFmpeg`` or ``AVBin``: For playing compressed videos.

To get started quickly, this should suffice for conda users on most systems:

.. code-block:: console

    $ conda create -n expy mne "pyglet<1.6" -c conda-forge
    $ conda activate expy
    $ pip install pyparallel rtmixer
    $ pip install git+https://github.com/labsn/expyfun

where ``expy`` can be replaced with whatever name you find convenient. If you
cannot (or don't want to) use conda-forge as a package source, you'll have to
do this instead:

.. code-block:: console

    $ conda create -n expy python=3 numpy scipy matplotlib pandas h5py joblib pillow
    $ conda activate expy
    $ pip install mne "pyglet<1.6" pyparallel rtmixer
    $ pip install git+https://github.com/labsn/expyfun

If you prefer using pip for everything, here are the minimum requirements:

.. code-block:: console

    $ pip install mne matplotlib "pyglet<1.6" pillow
    $ pip install git+https://github.com/labsn/expyfun

and this does a full pip install of all required and optional dependencies:

.. code-block:: console

    $ pip install mne matplotlib "pyglet<1.6" pillow
    $ pip install rtmixer pyparallel pandas joblib h5py TDTPy
    $ pip install git+https://github.com/labsn/expyfun

Note that the pyglet package for the recommended installs is constrained to version 1.5, as this
will be the last version compatible with legacy OpenGL (see pypi.org/project/pyglet/). If
you prefer to download pyglet via its github repository, please use the pyglet-1.5-maintenance
branch.


Expyfun
^^^^^^^
The recommended way to install expyfun on
development machines is to ``git clone`` the repository then do:

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
- parallel port breakout, TDT trigger breakout, or sound card SPDIF-to-TTL
  converter
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

- ``"SCREEN_SIZE_PIX"``
    Comma-separated full screen size in pixels, e.g., ``"1920,1200"``.
- ``"SCREEN_DISTANCE"``
    Physical display distance from the subject, e.g., ``"83.0"``.
- ``"SCREEN_WIDTH"``
    Physical display width, e.g., ``"52.0"``.

Another settable parameter is ``"SCREEN_HEIGHT"``, but if you have square
display pixels (a sane assumption for reasonable displays) then it's inferred
based on the screen size in pixels and physical screen width.

Other settings depend on whether you use TDT / sound card / parallel port for
auditory stimuli and triggering. Possibilities can be seen by looking at
:obj:`expyfun.known_config_types`. Your current system configuration can be
viewed by doing::

    >>> expyfun.get_config()
    {'SCREEN_DISTANCE': '61.0', 'SCREEN_SIZE_PIX': '1920,1200', 'SCREEN_WIDTH': '52.0', 'SOUND_CARD_BACKEND': 'rtmixer'}

.. note::

    If this returns ``{}``, you have not written any config values yet. This
    means that the standard ``expyfun.json`` file might not exist, and
    you might want to do something like::

        >>> expyfun.set_config('SCREEN_SIZE_PIX', '1920,1200')

    To initialize the ``expyfun.json`` file.


The fixed, hardware-dependent settings for a given system get written to
an ``expyfun.json`` file. You can use :func:`expyfun.get_config_path` to
get the path to your config file. Some sample configurations:

A TDT-based M/EEG+pupillometry machine
  .. code-block:: JSON

    {
    "AUDIO_CONTROLLER": "tdt",
    "EXPYFUN_EYELINK": "100.1.1.1",
    "RESPONSE_DEVICE": "keyboard",
    "SCREEN_DISTANCE": "100",
    "SCREEN_WIDTH": "51",
    "TDT_DELAY": "44",
    "TDT_INTERFACE": "GB",
    "TDT_MODEL": "RZ6",
    "TDT_TRIG_DELAY": "3",
    "TRIGGER_CONTROLLER": "tdt"
    }

A sound-card-based EEG system
  .. code-block:: JSON

    {
    "AUDIO_CONTROLLER": "sound_card",
    "RESPONSE_DEVICE": "keyboard",
    "SCREEN_DISTANCE": "50",
    "SCREEN_SIZE_PIX": "1920,1080",
    "SCREEN_WIDTH": "53",
    "SOUND_CARD_API": "ASIO",
    "SOUND_CARD_BACKEND": "rtmixer",
    "SOUND_CARD_FIXED_DELAY": 0.03,
    "SOUND_CARD_FS": 48000,
    "SOUND_CARD_NAME": "ASIO Fireface USB",
    "SOUND_CARD_TRIGGER_CHANNELS": 2,
    "TRIGGER_CONTROLLER": "sound_card"
    }

Deploying experiments
---------------------
The function :func:`expyfun.download_version` should be used to deploy a
static version of expyfun once an experiment is in its finalized state.
