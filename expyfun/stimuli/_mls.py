# -*- coding: utf-8 -*-
# Copyright (c) 2014, LABSN.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

"""Maximum-length sequence (MLS) impulse-response finding functions
"""

from os import path as op
import numpy as np
from scipy.fftpack import ifft, fft

from .._utils import verbose_dec, logger

_mls_file = op.join(op.dirname(__file__), '..', 'data', 'mls.bin')
_max_bits = 14  # determined by how the file was made, see _max_len_wrapper


def _check_n_bits(n_bits):
    """Helper to make sure we have a usable number of bits"""
    if not isinstance(n_bits, int):
        raise TypeError('n_bits must be an integer')
    if n_bits < 2 or n_bits > _max_bits:
        raise ValueError('n_bits must be between 2 and %s' % _max_bits)


def _max_len_wrapper(n_bits):
    """Maximum Length Sequence (MLS) generator

    Parameters
    ----------
    n_bits : int
        Number of bits to use. Length of the resulting sequence will
        be ``(2**n) - 1``. Only values between 2 and 15 supported.

    Returns
    -------
    seq : array
        Resulting MLS sequence of -1's and 1's.
    """
    n_bits = int(n_bits)
    _check_n_bits(n_bits)
    # This was used to generate the sequences:
    #from scipy.signal import max_len_seq
    #_mlss = np.concatenate([max_len_seq(n) > 0
    #                        for n in range(2, _max_bits + 1)])
    #with open(_mls_file, 'wb') as fid:
    #    fid.write(_mlss.tostring())
    _lims = np.cumsum([0] + [2 ** n - 1 for n in range(2, 15)])
    _mlss = np.fromfile(_mls_file, dtype=bool)
    _mlss = [_mlss[l1:l2].copy() for l1, l2 in zip(_lims[:-1], _lims[1:])]
    return _mlss[n_bits - 2] * 2. - 1


# Once this is in upstream scipy, we can add this:
#try:
#    from scipy.signal import max_len_seq as _max_len_seq
#except:
_max_len_seq = _max_len_wrapper


def repeated_mls(n_samp, n_repeats):
    """Generate a repeated MLS 0/1 signal for finding an impulse response

    Parameters
    ----------
    n_samp : int
        The estimated maximum number of samples in the impulse response.
    n_repeats : int
        The number of repeats to use.
    """
    if not isinstance(n_samp, int) or not isinstance(n_repeats, int):
        raise TypeError('n_samp and n_repeats must both be integers')
    n_bits = max(int(np.ceil(np.log2(n_samp + 1))), 2)
    if n_bits > _max_bits:
        raise ValueError('Only lengths up to %s supported'
                         % (2 ** _max_bits - 1))
    mls = 0.5 * _max_len_seq(n_bits) + 0.5
    n_resp = len(mls) * (n_repeats + 1) - 1
    mls = np.tile(mls, n_repeats)
    return mls, n_resp


@verbose_dec
def compute_mls_impulse_response(response, mls, n_repeats, verbose=None):
    """Compute the impulse response from data obtained using MLS

    Parameters
    ----------
    response : array
        Response of the system to the repeated MLS.
    mls : array
        The MLS presented to the system.
    n_repeats : int
        Number of repeats used.
    """
    if mls.ndim != 1 or response.ndim != 1:
        raise ValueError('response and mls must both be one-dimensional')
    if not isinstance(n_repeats, int):
        raise TypeError('n_repeats must be an integer')
    if not np.array_equal(np.sort(np.unique(mls)), [0, 1]):
        raise ValueError('MLS must be sequence of 0s and 1s')
    if mls.size % n_repeats != 0:
        raise ValueError('MLS length (%s) is not a multiple of the number '
                         'of repeats (%s)' % (mls.size, n_repeats))
    mls_len = mls.size // n_repeats
    n_bits = int(np.round(np.log2(mls_len + 1)))
    n_check = 2 ** n_bits
    if n_check != mls_len + 1:
        raise RuntimeError('length of MLS must be one shorter than a power '
                           'of 2, got %s (close to %s)' % (mls_len, n_check))
    logger.info('MLS using %s bits detected' % n_bits)
    n_len = response.size + 1
    if n_len % mls_len != 0:
        n_rep = int(np.round(n_len / float(mls_len)))
        n_len = mls_len * n_rep - 1
        raise ValueError('length of data must be one shorter than a '
                         'multiple of the MLS length (%s), found a length '
                         'of %s which is close to %s (%s repeats)'
                         % (mls_len, response.size, n_len, n_rep))
    # Now that we know our signal, we can actually deconvolve.
    # First, wrap the end back to the beginning
    resp_wrap = response[:n_repeats * mls_len].copy()
    resp_wrap[:mls_len - 1] += response[n_repeats * mls_len:]
    # Compute the circular crosscorrelation, w/correction for MLS scaling
    correction = np.empty(len(mls))
    correction.fill(1. / (2 ** (n_bits - 2) * n_repeats))
    correction[0] = 1. / ((4 ** (n_bits - 1)) * n_repeats)
    y = np.real(ifft(correction * fft(resp_wrap) * fft(mls).conj()))
    # Average out repeats
    h_est = np.mean(np.reshape(y, (n_repeats, mls_len)), axis=0)
    return h_est
