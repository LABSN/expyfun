# -*- coding: utf-8 -*-
"""HDF5 saving functions
"""

import numpy as np
from os import path as op

from .._utils import _check_pytables


def write_hdf5_dict(fname, data, overwrite=False):
        """Save dictionary to HDF5 format using Pytables

        Parameters
        ----------
        fname : str
            Filename to use.
        data : dict
            Dictionary to write. Can contain fields that are of the following
            types:
                {ndarray, dict}
        overwrite : bool
            If True, overwrite file (if it exists).
        """
        tb = _check_pytables()
        if op.isfile(fname) and not overwrite:
            raise IOError('file "%s" exists, use overwrite=True to overwrite'
                          % fname)
        if not isinstance(data, dict):
            raise TypeError('data must be a dict')
        o_f = tb.open_file if hasattr(tb, 'open_file') else tb.openFile
        with o_f(fname, mode='w') as fid:
            if hasattr(fid, 'create_group'):
                c_g = fid.create_group
                c_c_a = fid.create_carray
            else:
                c_g = fid.createGroup
                c_c_a = fid.createCArray
            filters = tb.Filters(complib='zlib', complevel=5)
            _save_dict(fid.root, data, filters, c_g, c_c_a)


def _save_dict(root, data, filters, c_g, c_c_a):
    """Helper to add a dict node"""
    assert isinstance(data, dict)
    tb = _check_pytables()
    for key, value in data.items():
        if isinstance(value, dict):
            sub_root = c_g(root, key)
            _save_dict(sub_root, value, filters, c_g, c_c_a)
        elif isinstance(value, np.ndarray):
            atom = tb.Atom.from_dtype(value.dtype)
            s = c_c_a(root, key, atom,
                      value.shape, filters=filters)
            s[:] = value
        else:
            raise TypeError('unsupported type %s' % type(value))


def read_hdf5_dict(fname):
    """Load dictionary from HDF5 format using Pytables

    Parameters
    ----------
    fname : str
        File to load.

    Returns
    -------
    data : dict
        The loaded data.
    """
    tb = _check_pytables()
    if not op.isfile(fname):
        raise IOError('file "%s" not found' % fname)
    data = dict()
    o_f = tb.open_file if hasattr(tb, 'open_file') else tb.openFile
    with o_f(fname, mode='r') as fid:
        _load_dict(fid.root, data)
    return data


def _load_dict(root, data):
    """Helper to populate a given dictionary from subnodes"""
    tb = _check_pytables()
    for node in root:
        key = node._v_name
        if isinstance(node, tb.Group):
            data[key] = dict()
            _load_dict(node, data[key])
        else:
            data[key] = np.array(node)
