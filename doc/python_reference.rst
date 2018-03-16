=============
API Reference
=============

.. automodule:: expyfun
   :no-members:
   :no-inherited-members:

This is the classes and functions reference of expyfun. Functions are
grouped by hardware control type.


.. contents::
   :local:
   :depth: 2


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
   ParallelTrigger
   TDTController

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

Classes:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   CRMPreload
   TrackerBinom
   TrackerDealer
   TrackerUD

Functions:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   convolve_hrtf
   compute_mls_impulse_response
   crm_info
   crm_prepare_corpus
   crm_response_menu
   crm_sentence
   play_sound
   repeated_mls
   rms
   texture_ERB
   vocode
   window_edges

:py:mod:`expyfun.io`:

.. currentmodule:: expyfun.io

Functions:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   read_hdf5
   read_tab_raw
   read_tab
   read_wav
   write_hdf5
   write_wav
   reconstruct_tracker
   reconstruct_dealer

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
   box_off
   dprime
   dprime_2afc
   fit_sigmoid
   format_pval
   logit
   plot_screen
   press_times_to_hmfc
   restore_values
   rt_chisq
   sigmoid

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
