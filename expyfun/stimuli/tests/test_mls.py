import numpy as np
import pytest
from numpy.testing import assert_allclose

from expyfun.stimuli import compute_mls_impulse_response, repeated_mls


def test_mls_ir():
    """Test computing impulse response with MLS"""
    # test simple stuff
    for _ in range(5):
        # make sure our signals have some DC
        sig_len = np.random.randint(10, 2000)
        kernel = np.random.rand(sig_len) + 10 * np.random.rand(1)
        n_repeats = 10

        mls, n_resp = repeated_mls(len(kernel), n_repeats)
        resp = np.zeros(n_resp)
        resp[: len(mls) + len(kernel) - 1] = np.convolve(mls, kernel)

        est_kernel = compute_mls_impulse_response(resp, mls, n_repeats)
        kernel_pad = np.zeros(len(est_kernel))
        kernel_pad[: len(kernel)] = kernel
        assert_allclose(kernel_pad, est_kernel, atol=1e-5, rtol=1e-5)

    # failure modes
    pytest.raises(TypeError, repeated_mls, "foo", n_repeats)
    pytest.raises(ValueError, compute_mls_impulse_response, resp[:-1], mls, n_repeats)
    pytest.raises(ValueError, compute_mls_impulse_response, resp, mls[:-1], n_repeats)
    pytest.raises(
        ValueError, compute_mls_impulse_response, resp, mls * 2.0 - 1.0, n_repeats
    )
    pytest.raises(
        ValueError, compute_mls_impulse_response, resp, mls[np.newaxis, :], n_repeats
    )
