# -*- coding: utf-8 -*-
from os import path as op
from nose.tools import assert_raises, assert_true

import numpy as np

from expyfun.io import write_hdf5, read_hdf5
from expyfun._utils import _TempDir, object_diff, requires_pytables

tempdir = _TempDir()


@requires_pytables
def test_hdf5():
    """Test HDF5 IO
    """
    test_file = op.join(tempdir, 'test.hdf5')
    x = dict(a=dict(b=np.zeros(3)), c=np.zeros(2, np.complex256),
             d=[dict(e=(1, -2., 'hello'))])
    assert_raises(TypeError, write_hdf5, test_file, 1)
    write_hdf5(test_file, x)
    assert_raises(IOError, write_hdf5, test_file, x)  # file exists
    write_hdf5(test_file, x, overwrite=True)
    assert_raises(IOError, read_hdf5, test_file + 'FOO')  # not found
    xx = read_hdf5(test_file)
    print(object_diff(x, xx))
    assert_true(object_diff(x, xx) == '')  # no assert_equal, ugly output
