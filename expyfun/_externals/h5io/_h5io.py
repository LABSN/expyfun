# -*- coding: utf-8 -*-
"""HDF5 saving functions
"""

import numpy as np
from os import path as op

from .._utils import _check_h5py, string_types, text_type


##############################################################################
# WRITE

def _create_titled_group(root, key, title):
    """Helper to create a titled group in h5py"""
    out = root.create_group(key)
    out.attrs['TITLE'] = title
    return out


def _create_titled_dataset(root, key, title, data, comp_kw=None):
    """Helper to create a titled dataset in h5py"""
    comp_kw = {} if comp_kw is None else comp_kw
    out = root.create_dataset(key, data=data, **comp_kw)
    out.attrs['TITLE'] = title
    return out


def write_hdf5(fname, data, overwrite=False, compression=5):
    """Write python object to HDF5 format using h5py

    Parameters
    ----------
    fname : str
        Filename to use.
    data : object
        Object to write. Can be of any of these types:
            {ndarray, dict, list, tuple, int, float, str}
        Note that dict objects must only have ``str`` keys.
    overwrite : bool
        If True, overwrite file (if it exists).
    compression : int
        An integer between 0 (no compression) to 9 (highest compression)
        using zlib.
    """
    h5py = _check_h5py()
    compression = int(compression)
    if compression < 0 or compression > 9:
        raise ValueError('Compression must be between 0 and 9')
    if op.isfile(fname) and not overwrite:
        raise IOError('file "%s" exists, use overwrite=True to overwrite'
                      % fname)
    comp_kw = dict()
    if compression > 0:
        comp_kw = dict(compression='gzip', compression_opts=compression)
    with h5py.File(fname, mode='w') as fid:
        _triage_write('expyfun', data, fid, comp_kw)


def _triage_write(key, value, root, comp_kw):
    if isinstance(value, dict):
        sub_root = _create_titled_group(root, key, 'dict')
        for key, sub_value in value.items():
            if not isinstance(key, string_types):
                raise TypeError('All dict keys must be strings')
            _triage_write('key_{0}'.format(key), sub_value, sub_root, comp_kw)
    elif isinstance(value, (list, tuple)):
        title = 'list' if isinstance(value, list) else 'tuple'
        sub_root = _create_titled_group(root, key, title)
        for vi, sub_value in enumerate(value):
            _triage_write('idx_{0}'.format(vi), sub_value, sub_root, comp_kw)
    elif isinstance(value, type(None)):
        _create_titled_dataset(root, key, 'None', [False])
    elif isinstance(value, (int, float)):
        if isinstance(value, int):
            title = 'int'
        else:  # isinstance(value, float):
            title = 'float'
        _create_titled_dataset(root, key, title, np.atleast_1d(value))
    elif isinstance(value, string_types):
        if isinstance(value, text_type):  # unicode
            value = np.fromstring(value.encode('utf-8'), np.uint8)
            title = 'unicode'
        else:
            value = np.fromstring(value.encode('ASCII'), np.uint8)
            title = 'ascii'
        _create_titled_dataset(root, key, title, value, comp_kw)
    elif isinstance(value, np.ndarray):
        _create_titled_dataset(root, key, 'ndarray', value)
    else:
        raise TypeError('unsupported type %s' % type(value))


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
    h5py = _check_h5py()
    if not op.isfile(fname):
        raise IOError('file "%s" not found' % fname)
    with h5py.File(fname, mode='r') as fid:
        if 'expyfun' not in fid.keys():
            raise TypeError('no expyfun data found')
        data = _triage_read(fid['expyfun'])
    return data


def _triage_read(node):
    h5py = _check_h5py()
    type_str = node.attrs['TITLE']
    if isinstance(type_str, bytes):
        type_str = type_str.decode()
    if isinstance(node, h5py.Group):
        if type_str == 'dict':
            data = dict()
            for key, subnode in node.items():
                data[key[4:]] = _triage_read(subnode)
        elif type_str in ['list', 'tuple']:
            data = list()
            ii = 0
            while True:
                subnode = node.get('idx_{0}'.format(ii), None)
                if subnode is None:
                    break
                data.append(_triage_read(subnode))
                ii += 1
            assert len(data) == ii
            data = tuple(data) if type_str == 'tuple' else data
            return data
        else:
            raise NotImplementedError('Unknown group type: {0}'
                                      ''.format(type_str))
    elif type_str == 'ndarray':
        data = np.array(node)
    elif type_str in ('int', 'float'):
        if type_str == 'int':
            cast = int
        else:  # type_str == 'float':
            cast = float
        data = cast(np.array(node)[0])
    elif type_str in ('unicode', 'ascii', 'str'):  # 'str' for backward compat
        decoder = 'utf-8' if type_str == 'unicode' else 'ASCII'
        cast = text_type if type_str == 'unicode' else str
        data = cast(np.array(node).tostring().decode(decoder))
    elif type_str == 'None':
        data = None
    else:
        raise TypeError('Unknown node type: {0}'.format(type_str))
    return data
