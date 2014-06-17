# -*- coding: utf-8 -*-
from os import path as op
from nose.tools import assert_raises, assert_in, assert_not_in

import numpy as np

from expyfun.io import write_hdf5_dict, read_hdf5_dict
from expyfun._utils import _TempDir

tempdir = _TempDir()


def test_hdf5():
    """Test HDF5 IO
    """
    test_file = op.join(tempdir, 'test.hdf5')
    x = dict(w=dict(y=np.zeros(3)), z=np.zeros(2, np.complex256))
    assert_raises(TypeError, write_hdf5_dict, test_file, 1)
    write_hdf5_dict(test_file, x)
    assert_raises(IOError, write_hdf5_dict, test_file, x)  # file exists
    write_hdf5_dict(test_file, x, overwrite=True)
    assert_raises(IOError, read_hdf5_dict, test_file + 'FOO')  # not found
    xx = read_hdf5_dict(test_file)
    for k in ['w', 'z']:
        assert_in(k, list(xx.keys()))
    assert_not_in('y', list(xx.keys()))
