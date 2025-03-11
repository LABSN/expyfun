:orphan:

.. _parallel_installation:

Parallel port triggering
========================

.. highlight:: console

A PCIe device such as "StarTech.com 1 Port PCI Express" should work.
USB to parallel port adapters will not, due to hardware limitations of
the devices ($10 ones on Amazon are only designed for printers) and the
USB protocol itself, which is not designed for low-latency control.

Instructions differ between Linux and Windows:

Linux
    On Linux, you need ``pyparallel``::

        $ pip install pyparallel

    You might also need some combination of the following:

    1. ``$ sudo modprobe ppdev``
    2. Add user to lp group (``/etc/group``)
    3. Run ``sudo rmmod lp`` (otherwise lp takes exclusive control)
    4. Edit ``/etc/modprobe.d/blacklist.conf`` to add blacklist ``lp``
    5. ``$ ls /dev/parport*`` to get the parallel port address, e.g.
       ``'/dev/parport0'``, and set this as ``TRIGGER_ADDRESS`` in the config.

Windows
    If you are on a modern Windows system (i.e., 64-bit), you'll need to:

    - Download the latest "binaries" archive from the `InpOut32 site`_
    - Extract the files
    - Run the ``Win32\InstallDriver.exe`` file (yes, even though it's in the
      Win32 directory)
    - Rename the **64-bit** file ``inpoutx64.dll`` to ``inpout32.dll``
    - Place this file in ``C:\Windows\System32\``
    - Use the Device Manager (or some other method) to get the parallel port
      address (from Ports➡Properties➡Resources➡I/O Range), e.g. ``0x378``
      or ``0xCFF4``, and set this as ``TRIGGER_ADDRESS`` in the config.
    - If you have trouble, you can interactively test your parallel port using
      the `parallel port tester application`_.

.. _`InpOut32 site`: http://www.highrez.co.uk/downloads/inpout32/
.. _`parallel port tester application`: https://www.downtowndougbrown.com/2013/06/parallel-port-tester/
