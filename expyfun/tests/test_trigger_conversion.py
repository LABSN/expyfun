from numpy.testing import assert_array_equal
from nose.tools import assert_raises

from expyfun import decimals_to_binary, binary_to_decimals


def test_conversion():
    """Test decimal<->binary conversion
    """
    assert_raises(ValueError, decimals_to_binary, [1], [0])
    assert_raises(ValueError, decimals_to_binary, [-1], [1])
    assert_raises(ValueError, decimals_to_binary, [1, 1], [1])
    assert_raises(ValueError, decimals_to_binary, [2], [1])
    assert_raises(ValueError, binary_to_decimals, [2], [1])
    assert_raises(ValueError, binary_to_decimals, [0.5], [1])
    assert_raises(ValueError, binary_to_decimals, [-1], [1])
    assert_raises(ValueError, binary_to_decimals, [[1]], [1])
    assert_raises(ValueError, binary_to_decimals, [1], [-1])
    assert_raises(ValueError, binary_to_decimals, [1], [2])
    # test cases
    decs = [[1],
            [1, 0, 1, 4, 5],
            [0, 3],
            [3, 0],
            ]
    bits = [[1],
            [1, 1, 2, 4, 4],
            [2, 2],
            [2, 2],
            ]
    bins = [[1],
            [1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            ]
    for d, n, b in zip(decs, bits, bins):
        assert_array_equal(decimals_to_binary(d, n), b)
        assert_array_equal(binary_to_decimals(b, n), d)
