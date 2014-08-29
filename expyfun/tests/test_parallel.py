# -*- coding: utf-8 -*-

import numpy as np
import warnings
from nose.tools import assert_raises
from numpy.testing import assert_array_equal

from expyfun._parallel import parallel_func, _check_n_jobs
from expyfun._utils import requires_joblib

warnings.simplefilter('always')


def fun(x):
    return x


@requires_joblib
def test_parallel():
    """Test parallel support
    """
    assert_raises(TypeError, _check_n_jobs, 'foo')
    parallel, p_fun, _ = parallel_func(fun, 1)
    a = np.array(parallel(p_fun(x) for x in range(10)))
    parallel, p_fun, _ = parallel_func(fun, 2)
    b = np.array(parallel(p_fun(x) for x in range(10)))
    assert_array_equal(a, b)
