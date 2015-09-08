# -*- coding: utf-8 -*-
from ._wav import read_wav, write_wav
from .._externals._h5io import (read_hdf5 as _read_hdf5,
                                write_hdf5 as _write_hdf5)
from ._parse import read_tab


def read_hdf5(fname):
    """..."""
    return _read_hdf5(fname, title='expyfun')


def write_hdf5(fname, data, overwrite=False, compression=4):
    """..."""
    return _write_hdf5(fname, data, overwrite, compression, title='expyfun')
