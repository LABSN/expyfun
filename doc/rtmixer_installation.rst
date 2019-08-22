:orphan:

.. _rtmixer_installation:

Installing rtmixer
==================

:mod:`rtmixer` seems to be best choice for sound playback on Python, as it
aims to guarantee a fixed timing delay (near-zero jitter) and no problems with
the Python global interpreter lock.

However, it comes with somewhat challenging installation requirements at the
moment. It requires both PortAudio to be installed, and the Python library
itself to be compiled from source (no pip wheels yet).

Here are some stub installation instructions for each OS:

- |linux| Linux / |apple| macOS
    .. code-block:: console

       $ sudo apt install libportaudio2  # only if on Linux!
       $ pip install sounddevice
       $ python -m sounddevice  # just to see if it worked
       $ git clone git://github.com/spatialaudio/python-rtmixer
       $ cd python-rtmixer
       $ git submodule update --init
       $ pip install -e .


    .. note:: As an alternative to ``brew`` on macOS, you can also use the
             `PortAudio binaries site`_ to put the ``dylib`` in some
             suitable macOS library path.

- |windows| Windows

    Because Windows support for compiling is pretty bad, ``python-rtmixer``
    probably will not install directly using ``pip``, because you need
    Visual Studio (typically version 14.0+ should work). If you are running
    Python 3.7, you can use this precompiled binary for now:

    .. code-block:: console

       $ pip install sounddevice
       $ python -m sounddevice  # just to see if it worked
       $ pip install https://staff.washington.edu/larsoner/rtmixer-0.0.0-cp37-cp37m-win_amd64.whl

    If you are on Python 3.6 you can try
    `this binary <https://staff.washington.edu/larsoner/rtmixer-0.0.0-cp36-cp36m-win_amd64.whl>`__.

.. _`PortAudio binaries site`: https://github.com/spatialaudio/portaudio-binaries
