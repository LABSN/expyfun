=============
API Reference
=============

.. automodule:: expyfun
   :no-members:
   :no-inherited-members:

This is the classes and functions reference of expyfun. Functions are
grouped by hardware control type.


.. toctree::
   :maxdepth: 2

   python_reference


Experiment control
==================

:py:mod:`expyfun`:

.. currentmodule:: expyfun

Classes:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   ExperimentController
   EyelinkController

Functions:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   assert_version
   binary_to_decimals
   decimals_to_binary
   download_version
   get_keyboard_input
   wait_secs

Stimulus design
===============

:py:mod:`expyfun.stimuli`:

.. currentmodule:: expyfun.stimuli

Functions:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   convolve_hrtf
   compute_mls_impulse_response
   play_sound
   repeated_mls
   rms
   vocode
   window_edges

:py:mod:`expyfun.io`:

.. currentmodule:: expyfun.io

Functions:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   read_tab
   read_wav
   write_wav

:py:mod:`expyfun._externals.h5io`

.. currentmodule:: expyfun._externals.h5io

Functions

.. autosummary::
   :toctree: generated/
   :template: function.rst

   read_hdf5
   write_hdf5

:py:mod:`expyfun.visual`:

.. currentmodule:: expyfun.visual

Classes:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Circle
   ConcentricCircles
   Diamond
   FixationDot
   Line
   RawImage
   Rectangle
   Text

Code blocks
===========

:py:mod:`expyfun.codeblocks`:

.. currentmodule:: expyfun.codeblocks

Functions:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   find_pupil_dynamic_range
   find_pupil_tone_impulse_response

Analysis
========

:py:mod:`expyfun.analyze`:

.. currentmodule:: expyfun.analyze

Functions:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   barplot
   dprime
   dprime_2afc
   plot_screen
   restore_values

Logging and configuration
=========================

:py:mod:`expyfun`:

.. currentmodule:: expyfun

Functions:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   set_log_level
   set_config
