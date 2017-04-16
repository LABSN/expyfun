What's new
==========

.. _changes_4_0_dev0:

Current
-----------

Changelog
~~~~~~~~~



BUG
~~~

   - change description by `Author`_.

API
~~~

   - API change description.


.. _changes_3_0_0:

Version 3.0.0
-------------

Changelog
~~~~~~~~~

   - Support for parallelization via ``joblib``.
   - New stimulus generation methods for vocoding.
   - New response method: mouse position / mouse clicks.
   - New visual classes ``RawImage``, ``Triangle``, ``Diamond``, ``ConcentricCircles`` and ``FixationDot``.
   - New ``assert_version`` function allows experiment scripts to be tied to specific commit hashes.
   - Improved ``hdf5`` input/output using ``expyfun.io.read_hdf5`` and ``write_hdf5``.
   - Inline formatting of screen text via pyglet's ``decode_attributed`` markup scheme.
   - Support for Cedrus button boxes.
   - Trial sequence safeguard functions added: ``identify_trial`` must proceed ``start_stimulus`` which must proceed ``trial_ok``.
   - Exposed trial metadata through properties ``ExperimentController.participant``, ``session``, ``exp_name``, and ``data_fname``.
   - New EyeLink "dummy mode" for smoother testing when EyeLink not available.
   - Parsing function ``read_tab`` for collapsing ``ExperimentController`` output to wide-format CSV.
   - New conversion functions ``decimals_to_binary`` and ``binary_to_decimals``.
   - New analysis functions ``object_diff``, ``press_times_to_hmfc``, and ``rt_chiqsq``.
   - Improved ``barplot`` function, including better bracket collision avoidance and support for various p-value formatting schemes.
   - New codeblocks for finding pupil dilation impulse response.
   - Integration of HRTFs from CIPIC and BU databases for simulated spatial origins.

API changes summary
~~~~~~~~~~~~~~~~~~~

Here are the code migration instructions when upgrading from expyfun
version 2.0.0:

  - The biggest difference is the required use of ``identify_trial``, ``start_stimulus``, and ``trial_ok`` (in that order) in all scripts using `ExperimentController`. See the example scripts and documentation for further guidance.
  - Removed support for stamping via parallel port (insufficient testing).

Authors
~~~~~~~~~

The committer list for this release is the following (preceded by number
of commits):

    * 223 Eric Larson
    *  88  Daniel McCloy
    *  72  Ross Maddox
    *   6  Lindsey Kishline
    *   3  Dean Pospisil
    *   3  Mark Wronkewicz

.. _Eric Larson: http://faculty.washington.edu/larsoner/

.. _Daniel McCloy: http://dan.mccloy.info/

.. _Ross Maddox: http://faculty.washington.edu/rkmaddox/

.. _Lindsey Kishline: http://ilabs.washington.edu/research-staff/bio/i-labs-lindsey-kishline

.. _Dean Pospisil: http://ilabs.washington.edu/research-staff/bio/i-labs-dean-pospisil

.. _Mark Wronkewicz: http://ilabs.washington.edu/graduate-students/bio/i-labs-mark-wronkiewicz
