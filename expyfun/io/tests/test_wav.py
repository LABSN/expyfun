# -*- coding: utf-8 -*-
import numpy as np
import pytest
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_equal)
from os import path as op
import warnings

from expyfun._utils import _TempDir, _has_scipy_version
from expyfun.io import read_wav, write_wav

warnings.simplefilter('always')
tempdir = _TempDir()


def test_read_write_wav():
    """Test reading and writing WAV files
    """
    fname = op.join(tempdir, 'temp.wav')
    data = np.r_[np.random.rand(1000), 1, -1]
    fs = 44100

    # Use normal 16-bit precision: not great
    write_wav(fname, data, fs)
    data_read, fs_read = read_wav(fname)
    assert_equal(fs_read, fs)
    assert_array_almost_equal(data[np.newaxis, :], data_read, 4)

    # test our overwrite check
    pytest.raises(IOError, write_wav, fname, data, fs)

    # test forcing fs dtype to int
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        write_wav(fname, data, float(fs), overwrite=True)
        assert_equal(len(w), 1)

    # Use 64-bit int: not supported
    pytest.raises(RuntimeError, write_wav, fname, data, fs, dtype=np.int64,
                  overwrite=True)

    # Use 32-bit int: better
    write_wav(fname, data, fs, dtype=np.int32, overwrite=True)
    data_read, fs_read = read_wav(fname)
    assert_equal(fs_read, fs)
    assert_array_almost_equal(data[np.newaxis, :], data_read, 7)

    if _has_scipy_version('0.13'):
        # Use 32-bit float: better
        write_wav(fname, data, fs, dtype=np.float32, overwrite=True)
        data_read, fs_read = read_wav(fname)
        assert_equal(fs_read, fs)
        assert_array_almost_equal(data[np.newaxis, :], data_read, 7)

        # Use 64-bit float: perfect
        write_wav(fname, data, fs, dtype=np.float64, overwrite=True)
        data_read, fs_read = read_wav(fname)
        assert_equal(fs_read, fs)
        assert_array_equal(data[np.newaxis, :], data_read)
    else:
        pytest.raises(RuntimeError, write_wav, fname, data, fs,
                      dtype=np.float32, overwrite=True)

    # Now try multi-dimensional data
    data = np.tile(data[np.newaxis, :], (2, 1))
    write_wav(fname, data[np.newaxis, :], fs, overwrite=True)
    data_read, fs_read = read_wav(fname)
    assert_equal(fs_read, fs)
    assert_array_almost_equal(data, data_read, 4)

    # Make sure our bound check works
    pytest.raises(ValueError, write_wav, fname, data * 2, fs, overwrite=True)
