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

   wait_secs

Experiment Design
=================

.. currentmodule:: expyfun.stimuli

Functions:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   read_wav
   rms
   write_wav

.. currentmodule:: expyfun.visual

Classes:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Circle
   Line
   RawImage
   Rectangle
   Text

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

Logging and Configuration
=========================

.. currentmodule:: expyfun

Functions:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   set_log_level
   set_config
