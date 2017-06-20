# -*- coding: utf-8 -*-
from ._wav import read_wav, write_wav
from .._externals._h5io import (read_hdf5 as _read_hdf5,
                                write_hdf5 as _write_hdf5)
from ._parse import (read_tab, reconstruct_tracker, 
                     reconstruct_dealer, read_tab_raw)


def read_hdf5(fname):
    """Read python object from HDF5 format using h5io/h5py

    Parameters
    ----------
    fname : str
        File to load.

    Returns
    -------
    data : object
        The loaded data. Can be of any type supported by :func:`write_hdf5`.

    See Also
    --------
    write_hdf5
    """
    return _read_hdf5(fname, title='expyfun')


def write_hdf5(fname, data, overwrite=False, compression=4):
    """Write python object to HDF5 format using h5io/h5py

    Parameters
    ----------
    fname : str
        Filename to use.
    data : object
        Object to write. Can be of any of these types::

            {ndarray, dict, list, tuple, int, float, str}

        Note that dict objects must only have ``str`` keys.
    overwrite : bool
        If True, overwrite file (if it exists).
    compression : int
        Compression level to use (0-9) to compress data using gzip.

    See Also
    --------
    read_hdf5
    """
    return _write_hdf5(fname, data, overwrite, compression, title='expyfun')
