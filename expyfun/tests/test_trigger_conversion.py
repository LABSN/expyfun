from numpy.testing import assert_array_equal
from nose.tools import assert_raises

from expyfun import decimals_to_binary


def test_conversion():
    """Test decimal to binary conversion
    """
    assert_raises(ValueError, decimals_to_binary, [1], [0])
    assert_raises(ValueError, decimals_to_binary, [-1], [1])
    assert_raises(ValueError, decimals_to_binary, [1, 1], [1])
    assert_raises(ValueError, decimals_to_binary, [2], [1])
    assert_array_equal(decimals_to_binary([1], [1]), [1])
    assert_array_equal(decimals_to_binary([1, 0, 1, 4, 5], [1, 1, 2, 4, 4]),
                       [1,
                        0,
                        0, 1,
                        0, 1, 0, 0,
                        0, 1, 0, 1])
