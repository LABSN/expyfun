# -*- coding: utf-8 -*-

import numpy as np
import warnings
from nose.tools import assert_equal
from numpy.testing import assert_array_equal

from expyfun.stimuli import resample

warnings.simplefilter('always')


def test_resample():
    """Test resampling
    """
    x = np.random.normal(0, 1, (10, 10, 10))
    x_rs = resample(x, 1, 2, 10, n_jobs=2)
    assert_equal(x.shape, (10, 10, 10))
    assert_equal(x_rs.shape, (10, 10, 5))

    x_2 = x.swapaxes(0, 1)
    x_2_rs = resample(x_2, 1, 2, 10)
    assert_array_equal(x_2_rs.swapaxes(0, 1), x_rs)

    x_3 = x.swapaxes(0, 2)
    x_3_rs = resample(x_3, 1, 2, 10, 0)
    assert_array_equal(x_3_rs.swapaxes(0, 2), x_rs)
