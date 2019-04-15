=============
API Reference
=============

This is the classes and functions reference of expyfun. Functions are
grouped by hardware control type.

.. contents::
   :local:
   :depth: 2

.. currentmodule:: expyfun

.. automodule:: expyfun
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   ExperimentController
   EyelinkController
   ParallelTrigger
   SoundCardController
   TDTController
   assert_version
   binary_to_decimals
   check_units
   decimals_to_binary
   download_version
   get_config
   get_keyboard_input
   set_log_level
   set_config
   wait_secs


.. currentmodule:: expyfun.stimuli

.. automodule:: expyfun.stimuli
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   CRMPreload
   TrackerBinom
   TrackerDealer
   TrackerUD
   convolve_hrtf
   compute_mls_impulse_response
   crm_info
   crm_prepare_corpus
   crm_response_menu
   crm_sentence
   get_tdt_rates
   play_sound
   repeated_mls
   rms
   texture_ERB
   vocode
   window_edges


.. currentmodule:: expyfun.io

.. automodule:: expyfun.io
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   read_hdf5
   read_tab_raw
   read_tab
   read_wav
   write_hdf5
   write_wav
   reconstruct_tracker
   reconstruct_dealer


.. currentmodule:: expyfun.visual

.. automodule:: expyfun.visual
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   Circle
   ConcentricCircles
   Diamond
   FixationDot
   Line
   ProgressBar
   RawImage
   Rectangle
   Text
   Triangle
   Video


.. currentmodule:: expyfun.codeblocks

.. automodule:: expyfun.codeblocks
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   find_pupil_dynamic_range
   find_pupil_tone_impulse_response


:py:mod:`expyfun.analyze`:

.. currentmodule:: expyfun.analyze

.. automodule:: expyfun.analyze
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   barplot
   box_off
   dprime
   fit_sigmoid
   format_pval
   logit
   plot_screen
   press_times_to_hmfc
   restore_values
   rt_chisq
   sigmoid
