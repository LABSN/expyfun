# -*- coding: utf-8 -*-
"""HDF5 saving functions
"""

import numpy as np
from os import path as op

from .._utils import _check_pytables, string_types


##############################################################################
# WRITE

def write_hdf5_dict(fname, data, overwrite=False):
        """Write python object to HDF5 format using Pytables

        Parameters
        ----------
        fname : str
            Filename to use.
        data : object
            Object to write. Can be of any of these types:
                {ndarray, dict, list, tuple}
            Note that dict objects must only have ``str`` keys.
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
            _write_dict(fid.root, data, filters, c_g, c_c_a)


def _triage_write(key, value, root, filters, c_g, c_c_a):
    tb = _check_pytables()
    if isinstance(value, dict):
        sub_root = c_g(root, key, 'dict')
        _write_dict(sub_root, value, filters, c_g, c_c_a)
    elif isinstance(value, (list, tuple)):
        title = 'list' if isinstance(value, list) else 'tuple'
        sub_root = c_g(root, key, title)
        _write_list(sub_root, value, filters, c_g, c_c_a)
    elif isinstance(value, np.ndarray):
        atom = tb.Atom.from_dtype(value.dtype)
        s = c_c_a(root, key, atom,
                  value.shape, title='ndarray', filters=filters)
        s[:] = value
    else:
        raise TypeError('unsupported type %s' % type(value))


def _write_dict(root, data, filters, c_g, c_c_a):
    """Helper to add a dict node"""
    assert isinstance(data, dict)
    for key, value in data.items():
        if not isinstance(key, string_types):
            raise TypeError('All dict keys must be strings')
        _triage_write('key{0}'.format(key), value, root, filters, c_g, c_c_a)


def _write_list(root, data, filters, c_g, c_c_a):
    """Helper to add a dict node"""
    assert isinstance(data, (list, tuple))
    for vi, value in enumerate(data):
        _triage_write('idx{0}'.format(vi), value, root, filters, c_g, c_c_a)


##############################################################################
# READ

def read_hdf5(fname):
    """Read python object from HDF5 format using Pytables

    Parameters
    ----------
    fname : str
        File to load.

    Returns
    -------
    data : object
        The loaded data. Can be of any type supported by ``write_hdf5``.
    """
    tb = _check_pytables()
    if not op.isfile(fname):
        raise IOError('file "%s" not found' % fname)
    o_f = tb.open_file if hasattr(tb, 'open_file') else tb.openFile
    with o_f(fname, mode='r') as fid:
        data = _read_dict(fid.root)
    return data


def _triage_read(node):
    tb = _check_pytables()
    type_str = node._v_title
    if isinstance(node, tb.Group):
        if type_str == 'dict':
            data = _read_dict(node)
        elif type_str in ['list', 'tuple']:
            data = _read_list(node, type_str)
        else:
            raise NotImplementedError('Unknown group type: {0}'
                                      ''.format(type_str))
    else:
        if type_str != 'ndarray':
            raise TypeError('Unknown node type: {0}'.format(type_str))
        data = np.array(node)
    return data


def _read_dict(root):
    """Helper to populate a given dictionary from subnodes"""
    data = dict()
    for node in root:
        key = node._v_name[3:]  # cut off "idx" or "key" prefix
        data[key] = _triage_read(node)
    return data


def _read_list(root, type_):
    data = list()
    ii = 0
    while True:
        node = getattr(root, 'idx{0}'.format(ii), None)
        if node is None:
            break
        data.append(_triage_read(node))
        ii += 1
    return data
