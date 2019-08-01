# Prefer newest SciPy interface
try:
    from scipy.fft import rfft, irfft, rfftfreq  # noqa
except ImportError:
    from numpy.fft import rfft, irfft, rfftfreq  # noqa
