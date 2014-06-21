=========
Reference
=========

.. automodule:: expyfun
   :no-members:
   :no-inherited-members:

This is the classes and functions reference of expyfun. Functions are
grouped by hardware control type.


Experiment Control
==================

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

   get_keyboard_input
   wait_secs

Experiment Design
=================

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
   vocode_ci
   window_edges

.. currentmodule:: expyfun.io

Functions:

.. autosummary::
   read_hdf5
   read_wav
   write_hdf5
   write_wav

.. currentmodule:: expyfun.visual

Classes:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Circle
   ConcentricCircles
   FixationDot
   Line
   RawImage
   Rectangle
   Text

Experiment Code Blocks
======================

.. currentmodule:: expyfun.codeblocks

Functions:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   find_pupil_dynamic_range
   find_pupil_tone_impulse_response

Analysis
========

.. currentmodule:: expyfun.analyze

Functions:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   barplot
   dprime
   dprime_2afc
   plot_screen
   read_tab
   restore_values

Logging and Configuration
=========================

.. currentmodule:: expyfun

Functions:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   set_log_level
   set_config
