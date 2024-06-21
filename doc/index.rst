=======
expyfun
=======

.. rst-class:: h4 font-weight-light my-4

   A high-precision auditory and visual stimulus delivery library for
   psychoacoustics in Python.

Purpose
-------
Expyfun is desigend for use by a handful of labs (e.g., LABSN and ILABS at the
University of Washington, AMP-LAB at University of Rochester, etc.).
We welcome contributions from other labs if they serve this mission, but cannot
work to support use cases that don't fit those of the above labs at this time.

Design philosophy
-----------------
- Adherence to Python community coding standards
- Unit testing and continuous integration
- Tight revision / version control
- Limit hardware support to simplify support cases
- Allow paradigm *development* to occur on any machine, for eventual
  *deployment* on machines specifically dedicated to experiments

Hardware support
----------------
- Tucker-Davis Technologies (TDT) hardware (RP2, RZ6) for:

  - Precise auditory delivery
  - Trigger stamping for integration with eye tracker/EEG/MEG systems
  - TDT button box responses

- Sound card playback with:

  - Fixed delay (no jitter) relative to video flip on Linux and Windows
  - Digital trigger stamping for zero-jitter trigger-to-auditory delay
    (with custom digital-to-TTL hardware)

- EyeLink integration for pupillometry and eye tracking
- Parallel port triggering
- Keyboard responses
- Mouse responses
- Cedrus response boxes
- Joystick control / responses

.. toctree::
   :hidden:

   getting_started.rst
   python_reference.rst
   auto_examples/index.rst
