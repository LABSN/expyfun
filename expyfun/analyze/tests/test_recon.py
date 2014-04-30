import numpy as np
from numpy.testing import assert_allclose

from expyfun.analyze import restore_values


def test_restore():
    """Test restoring missing values
    """
    n = 20
    x = np.arange(n, dtype=float)
    y = x * 10 - 1.5
    keep = np.ones(n, bool)
    keep[[0, 4, -1]] = False
    missing = np.where(~keep)[0]
    keep = np.where(keep)[0]
    y = x[keep] * 10. - 1.5
    y2 = restore_values(x, y, missing)[0]
    x2 = (y2 + 1.5) / 10.
    assert_allclose(x, x2, atol=1e-7)
