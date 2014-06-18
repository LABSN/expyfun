"""Stimulus generation functions
"""

import numpy as np
import gzip

from ._filter import resample
from .._utils import fetch_data_file


# This was used to generate "barb_anech.gz":
#
# from scipy import io as spio
# mat = spio.loadmat('anechRev.mat')
# x = np.concatenate([mat[key][:, 0, -1, :].T.astype(np.float64)[:, None, :]
#                     for key in ['fimp_l', 'fimp_r']], axis=1)
# keep = np.abs(x) > np.max(np.abs(x)) * 1e-5
# idx = int(2 ** np.ceil(np.log2(np.where(np.any(np.any(keep, 1), 0))[0][-1])))
# x = x[:, :, :idx]
# sig = np.random.RandomState(0).randn(100000)
# res = np.array([np.convolve(sig, xx) for xx in x[0]])[:, idx:-idx]
# x /= np.mean(np.sqrt(np.mean(res ** 2, axis=1)))
# with gzip.open('barb_brir.gz', 'w') as fid:
#     fid.write(x.tostring())
# angles = np.arange(0, 91, 15, dtype=float)
# with gzip.open('barb_angles.gz', 'w') as fid:
#     fid.write(angles.tostring())
#
# Then the files were uploaded to lester.


def _get_hrtf(angle, source):
    """Helper to sub-select proper BRIR"""
    fnames = ['{0}_{1}.gz'.format(source, t) for t in ('angles', 'brir')]
    fnames = [fetch_data_file('hrtf/{0}'.format(fname))
              for fname in fnames]
    with gzip.open(fnames[0], 'r') as fid:
        angles = np.frombuffer(fid.read())
    leftward = False
    read_angle = angle
    if angle < 0:
        leftward = True
        read_angle = -angle
    if read_angle not in angles:
        raise ValueError('angle "{0}" must be one of +/-{1}'
                         ''.format(angle, list(angles)))
    with gzip.open(fnames[1], 'r') as fid:
        brir = np.frombuffer(fid.read())
    brir.shape = (angles.size, 2, brir.size // (2 * angles.size))
    idx = np.where(angles == read_angle)[0]
    assert len(idx) == 1
    brir = brir[idx[0]].copy()
    return brir, 44100, leftward


def convolve_hrtf(data, fs, angle, source='barb'):
    """Convolve a signal with a head-related transfer function

    Technically we will be convolving with binaural room impluse
    responses (BRIRs), but HRTFs (freq-domain equiv. representations)
    are the common terminology.

    Parameters
    ----------
    data : 1-dimensional array-like
        Data to operate on.
    fs : float
        The sample rate of the data. (HRTFs will be resampled if necessary.)
    angle : float
        The azimuthal angle of the HRTF.
    source : str
        Source to use for HRTFs. Currently only 'barb' is supported.

    Returns
    -------
    data_hrtf : array
        A 2D array ``shape=(2, n_samples)`` containing the convolved data.
    """
    fs = float(fs)
    angle = float(angle)
    known_sources = ['barb']
    if source not in known_sources:
        raise ValueError('Source "{0}" unknown, must be one of {1}'
                         ''.format(source, known_sources))
    data = np.array(data, np.float64)
    if data.ndim != 1:
        raise ValueError('data must be 1-dimensional')

    brir, brir_fs, leftward = _get_hrtf(angle, 'barb')
    order = [1, 0] if leftward else [0, 1]
    if not np.allclose(brir_fs, fs, rtol=0, atol=0.5):
        brir = [resample(b, fs, brir_fs) for b in brir]
    out = np.array([np.convolve(data, brir[o]) for o in order])
    return out
