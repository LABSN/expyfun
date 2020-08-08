"""Some utility functions"""

# Authors: Eric Larson <larsoner@uw.edu>
#
# License: BSD (3-clause)

import warnings
import operator
from copy import deepcopy
import subprocess
import importlib
import os
import os.path as op
import inspect
import sys
import time
import tempfile
import traceback
import ssl
from shutil import rmtree
import atexit
import json
from functools import partial
from distutils.version import LooseVersion
import logging
import datetime
from timeit import default_timer as clock
from threading import Timer

import numpy as np
import scipy as sp

from ._externals import decorator

# set this first thing to make sure it "takes"
try:
    import pyglet
    pyglet.options['debug_gl'] = False
    del pyglet
except Exception:
    pass


# for py3k (eventually)
if sys.version.startswith('2'):
    string_types = basestring  # noqa
    input = raw_input  # noqa, input is raw_input in py3k
    text_type = unicode  # noqa
    from __builtin__ import reload
    from urllib2 import urlopen  # noqa
    from cStringIO import StringIO  # noqa
else:
    string_types = str
    text_type = str
    from urllib.request import urlopen
    input = input
    from io import StringIO  # noqa, analysis:ignore
    from importlib import reload  # noqa, analysis:ignore

###############################################################################
# LOGGING

EXP = 25
logging.addLevelName(EXP, 'EXP')


def exp(self, message, *args, **kwargs):
    """Experiment-level logging."""
    self.log(EXP, message, *args, **kwargs)


logging.Logger.exp = exp
logger = logging.getLogger('expyfun')


def flush_logger():
    """Flush expyfun logger"""
    for handler in logger.handlers:
        handler.flush()


def set_log_level(verbose=None, return_old_level=False):
    """Convenience function for setting the logging level

    Parameters
    ----------
    verbose : bool, str, int, or None
        The verbosity of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL. Note that these are for
        convenience and are equivalent to passing in logging.DEBUG, etc.
        For bool, True is the same as 'INFO', False is the same as 'WARNING'.
        If None, the environment variable EXPYFUN_LOGGING_LEVEL is read, and if
        it doesn't exist, defaults to INFO.
    return_old_level : bool
        If True, return the old verbosity level.
    """
    if verbose is None:
        verbose = get_config('EXPYFUN_LOGGING_LEVEL', 'INFO')
    elif isinstance(verbose, bool):
        verbose = 'INFO' if verbose is True else 'WARNING'
    if isinstance(verbose, string_types):
        verbose = verbose.upper()
        logging_types = dict(DEBUG=logging.DEBUG, INFO=logging.INFO,
                             WARNING=logging.WARNING, ERROR=logging.ERROR,
                             CRITICAL=logging.CRITICAL)
        if verbose not in logging_types:
            raise ValueError('verbose must be of a valid type')
        verbose = logging_types[verbose]

    old_verbose = logger.level
    logger.setLevel(verbose)
    return (old_verbose if return_old_level else None)


def set_log_file(fname=None,
                 output_format='%(asctime)s - %(levelname)-7s - %(message)s',
                 overwrite=None):
    """Convenience function for setting the log to print to a file

    Parameters
    ----------
    fname : str, or None
        Filename of the log to print to. If None, stdout is used.
        To suppress log outputs, use set_log_level('WARN').
    output_format : str
        Format of the output messages. See the following for examples:
            http://docs.python.org/dev/howto/logging.html
        e.g., "%(asctime)s - %(levelname)s - %(message)s".
    overwrite : bool, or None
        Overwrite the log file (if it exists). Otherwise, statements
        will be appended to the log (default). None is the same as False,
        but additionally raises a warning to notify the user that log
        entries will be appended.
    """
    handlers = logger.handlers
    for h in handlers:
        if isinstance(h, logging.FileHandler):
            h.close()
        logger.removeHandler(h)
    if fname is not None:
        if op.isfile(fname) and overwrite is None:
            warnings.warn('Log entries will be appended to the file. Use '
                          'overwrite=False to avoid this message in the '
                          'future.')
        mode = 'w' if overwrite is True else 'a'
        lh = logging.FileHandler(fname, mode=mode)
    else:
        """ we should just be able to do:
                lh = logging.StreamHandler(sys.stdout)
            but because doctests uses some magic on stdout, we have to do this:
        """
        lh = logging.StreamHandler(WrapStdOut())

    lh.setFormatter(logging.Formatter(output_format))
    # actually add the stream handler
    logger.addHandler(lh)


###############################################################################
# RANDOM UTILITIES

building_doc = any('sphinx-build' in ((''.join(i[4]).lower() + i[1])
                                      if i[4] is not None else '')
                   for i in inspect.stack())


def run_subprocess(command, **kwargs):
    """Run command using subprocess.Popen

    Run command and wait for command to complete. If the return code was zero
    then return, otherwise raise CalledProcessError.
    By default, this will also add stdout= and stderr=subproces.PIPE
    to the call to Popen to suppress printing to the terminal.

    Parameters
    ----------
    command : list of str
        Command to run as subprocess (see subprocess.Popen documentation).
    **kwargs : objects
        Keywoard arguments to pass to ``subprocess.Popen``.

    Returns
    -------
    stdout : str
        Stdout returned by the process.
    stderr : str
        Stderr returned by the process.
    """
    # code adapted with permission from mne-python
    kw = dict(stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    kw.update(kwargs)

    p = subprocess.Popen(command, **kw)
    stdout_, stderr = p.communicate()

    output = (stdout_.decode(), stderr.decode())
    if p.returncode:
        err_fun = subprocess.CalledProcessError.__init__
        if 'output' in _get_args(err_fun):
            raise subprocess.CalledProcessError(p.returncode, command, output)
        else:
            raise subprocess.CalledProcessError(p.returncode, command)

    return output


class ZeroClock(object):
    """Clock that uses "clock" function but starts at zero on init."""

    def __init__(self):
        self._start_time = clock()

    def get_time(self):
        """Get time."""
        return clock() - self._start_time


def date_str():
    """Produce a date string for the current date and time

    Returns
    -------
    datestr : str
        The date string.
    """
    return str(datetime.datetime.today()).replace(':', '_')


class WrapStdOut(object):
    """Ridiculous class to work around how doctest captures stdout."""

    def __getattr__(self, name):
        # Even more ridiculous than this class, this must be sys.stdout (not
        # just stdout) in order for this to work (tested on OSX and Linux)
        return getattr(sys.stdout, name)


class _TempDir(str):
    """Class for creating and auto-destroying temp dir

    This is designed to be used with testing modules.

    We cannot simply use __del__() method for cleanup here because the rmtree
    function may be cleaned up before this object, so we use the atexit module
    instead. Passing del_after and print_del kwargs to the constructor are
    helpful primarily for debugging purposes.
    """

    def __new__(self, del_after=True, print_del=False):
        new = str.__new__(self, tempfile.mkdtemp())
        self._del_after = del_after
        self._print_del = print_del
        return new

    def __init__(self):
        self._path = self.__str__()
        atexit.register(self.cleanup)

    def cleanup(self):
        if self._del_after is True:
            if self._print_del is True:
                print('Deleting {} ...'.format(self._path))
            rmtree(self._path, ignore_errors=True)


def check_units(units):
    """Ensure user passed valid units type

    Parameters
    ----------
    units : str
        Must be ``'norm'``, ``'deg'``, ``'pix'``, or ``'cm'``.
    """
    good_units = ['norm', 'pix', 'deg', 'cm']
    if units not in good_units:
        raise ValueError('"units" must be one of {}, not {}'
                         ''.format(good_units, units))


###############################################################################
# DECORATORS

# Following deprecated class copied from scikit-learn

class deprecated(object):
    """Decorator to mark a function or class as deprecated.

    Issue a warning when the function is called/the class is instantiated and
    adds a warning to the docstring.

    The optional extra argument will be appended to the deprecation message
    and the docstring. Note: to use this with the default value for extra, put
    in an empty of parentheses:

    >>> from expyfun._utils import deprecated
    >>> deprecated() # doctest: +ELLIPSIS
    <expyfun._utils.deprecated object at ...>

    >>> @deprecated()
    ... def some_function(): pass
    """

    # Adapted from http://wiki.python.org/moin/PythonDecoratorLibrary,
    # but with many changes.

    # scikit-learn will not import on all platforms b/c it can be
    # sklearn or scikits.learn, so a self-contained example is used above

    def __init__(self, extra=''):
        """
        Parameters
        ----------
        extra: string
          to be added to the deprecation messages

        """
        self.extra = extra

    def __call__(self, obj):
        """Call."""
        if isinstance(obj, type):
            return self._decorate_class(obj)
        else:
            return self._decorate_fun(obj)

    def _decorate_class(self, cls):
        msg = "Class %s is deprecated" % cls.__name__
        if self.extra:
            msg += "; %s" % self.extra

        # FIXME: we should probably reset __new__ for full generality
        init = cls.__init__

        def wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return init(*args, **kwargs)
        cls.__init__ = wrapped

        wrapped.__name__ = '__init__'
        wrapped.__doc__ = self._update_doc(init.__doc__)
        wrapped.deprecated_original = init

        return cls

    def _decorate_fun(self, fun):
        """Decorate function fun"""
        msg = "Function %s is deprecated" % fun.__name__
        if self.extra:
            msg += "; %s" % self.extra

        def wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return fun(*args, **kwargs)

        wrapped.__name__ = fun.__name__
        wrapped.__dict__ = fun.__dict__
        wrapped.__doc__ = self._update_doc(fun.__doc__)

        return wrapped

    def _update_doc(self, olddoc):
        newdoc = "DEPRECATED"
        if self.extra:
            newdoc = "%s: %s" % (newdoc, self.extra)
        if olddoc:
            newdoc = "%s\n\n%s" % (newdoc, olddoc)
        return newdoc


if hasattr(inspect, 'signature'):  # py35
    def _get_args(function, varargs=False):
        params = inspect.signature(function).parameters
        args = [key for key, param in params.items()
                if param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)]
        if varargs:
            varargs = [param.name for param in params.values()
                       if param.kind == param.VAR_POSITIONAL]
            if len(varargs) == 0:
                varargs = None
            return args, varargs
        else:
            return args
else:
    def _get_args(function, varargs=False):
        out = inspect.getargspec(function)  # args, varargs, keywords, defaults
        if varargs:
            return out[:2]
        else:
            return out[0]


@decorator
def verbose_dec(function, *args, **kwargs):
    """Improved verbose decorator to allow functions to override log-level

    Do not call this directly to set global verbosrity level, instead use
    set_log_level().

    Parameters
    ----------
    function : callable
        Function to be decorated by setting the verbosity level.

    Returns
    -------
    dec - function
        The decorated function
    """
    arg_names = _get_args(function)

    if len(arg_names) > 0 and arg_names[0] == 'self':
        default_level = getattr(args[0], 'verbose', None)
    else:
        default_level = None

    if('verbose' in arg_names):
        verbose_level = args[arg_names.index('verbose')]
    else:
        verbose_level = default_level

    if verbose_level is not None:
        old_level = set_log_level(verbose_level, True)
        # set it back if we get an exception
        try:
            ret = function(*args, **kwargs)
        except Exception:
            set_log_level(old_level)
            raise
        set_log_level(old_level)
        return ret
    else:
        ret = function(*args, **kwargs)
        return ret


def _new_pyglet():
    import pyglet
    return LooseVersion(pyglet.version) >= LooseVersion('1.4')


def _has_video(raise_error=False):
    exceptions = list()
    good = True
    if _new_pyglet():
        try:
            from pyglet.media.codecs.ffmpeg import FFmpegSource  # noqa
        except ImportError:
            exceptions.append(traceback.format_exc())
            good = False
        else:
            if raise_error:
                print('Found FFmpegSource for new Pyglet')
    else:
        try:
            from pyglet.media.avbin import AVbinSource  # noqa
        except ImportError:
            exceptions.append(traceback.format_exc())
            try:
                from pyglet.media.sources.avbin import AVbinSource  # noqa
            except ImportError:
                exceptions.append(traceback.format_exc())
                good = False
            else:
                if raise_error:
                    print('Found AVbinSource for old Pyglet 1')
        else:
            if raise_error:
                print('Found AVbinSource for old Pyglet 2')
    if raise_error and not good:
        raise RuntimeError('Video support not enabled, got exception(s):\n\n%s'
                           '\n***********************\n'.join(exceptions))
    return good


def requires_video():
    """Require FFmpeg/AVbin."""
    import pytest
    return pytest.mark.skipif(not _has_video(), reason='Requires FFmpeg/AVbin')


def requires_opengl21(func):
    """Require OpenGL."""
    import pytest
    import pyglet.gl
    vendor = pyglet.gl.gl_info.get_vendor()
    version = pyglet.gl.gl_info.get_version()
    sufficient = pyglet.gl.gl_info.have_version(2, 0)
    return pytest.mark.skipif(not sufficient,
                              reason='OpenGL too old: %s %s'
                              % (vendor, version,))(func)


def requires_lib(lib):
    """Requires lib decorator."""
    import pytest
    try:
        importlib.import_module(lib)
    except Exception as exp:
        val = True
        reason = 'Needs %s (%s)' % (lib, exp)
    else:
        val = False
        reason = ''
    return pytest.mark.skipif(val, reason=reason)


def _has_scipy_version(version):
    return (LooseVersion(sp.__version__) >= LooseVersion(version))


def _get_user_home_path():
    """Return standard preferences path"""
    # this has been checked on OSX64, Linux64, and Win32
    val = os.getenv('APPDATA' if 'nt' == os.name.lower() else 'HOME', None)
    if val is None:
        raise ValueError('expyfun config file path could '
                         'not be determined, please report this '
                         'error to expyfun developers')
    return val


def fetch_data_file(fname):
    """Fetch example remote file

    Parameters
    ----------
    fname : str
        The remote filename to get. If the filename already exists
        on the local system, the file will not be fetched again.

    Returns
    -------
    fname : str
        The filename on the local system where the file was downloaded.
    """
    path = get_config('EXPYFUN_DATA_PATH', op.join(_get_user_home_path(),
                                                   '.expyfun', 'data'))
    fname_out = op.join(path, fname)
    if not op.isdir(op.dirname(fname_out)):
        os.makedirs(op.dirname(fname_out))
    fname_url = ('https://github.com/LABSN/expyfun-data/raw/master/{0}'
                 ''.format(fname))
    try:
        # until we get proper certificates
        context = ssl._create_unverified_context()
        this_urlopen = partial(urlopen, context=context)
    except AttributeError:
        context = None
        this_urlopen = urlopen
    if not op.isfile(fname_out):
        try:
            with open(fname_out, 'wb') as fid:
                www = this_urlopen(fname_url, timeout=30.0)
                try:
                    fid.write(www.read())
                finally:
                    www.close()
        except Exception:
            os.remove(fname_out)
            raise
    return fname_out


def get_config_path():
    r"""Get path to standard expyfun config file.

    Returns
    -------
    config_path : str
        The path to the expyfun configuration file. On windows, this
        will be '%APPDATA%\.expyfun\expyfun.json'. On every other
        system, this will be $HOME/.expyfun/expyfun.json.
    """
    val = op.join(_get_user_home_path(), '.expyfun', 'expyfun.json')
    return val


# List the known configuration values
known_config_types = ('RESPONSE_DEVICE',
                      'AUDIO_CONTROLLER',
                      'DB_OF_SINE_AT_1KHZ_1RMS',
                      'EXPYFUN_EYELINK',
                      'SOUND_CARD_API',
                      'SOUND_CARD_API_OPTIONS',
                      'SOUND_CARD_BACKEND',
                      'SOUND_CARD_FS',
                      'SOUND_CARD_NAME',
                      'SOUND_CARD_FIXED_DELAY',
                      'SOUND_CARD_TRIGGER_CHANNELS',
                      'SOUND_CARD_TRIGGER_INSERTION',
                      'SOUND_CARD_TRIGGER_SCALE',
                      'SOUND_CARD_TRIGGER_ID_AFTER_ONSET',
                      'TDT_CIRCUIT_PATH',
                      'TDT_DELAY',
                      'TDT_INTERFACE',
                      'TDT_MODEL',
                      'TDT_TRIG_DELAY',
                      'TRIGGER_CONTROLLER',
                      'TRIGGER_ADDRESS',
                      'WINDOW_SIZE',
                      'SCREEN_NUM',
                      'SCREEN_WIDTH',
                      'SCREEN_DISTANCE',
                      'SCREEN_SIZE_PIX',
                      'EXPYFUN_LOGGING_LEVEL',
                      )

# These allow for partial matches: 'NAME_1' is okay key if 'NAME' is listed
known_config_wildcards = ()


def get_config(key=None, default=None, raise_error=False):
    """Read expyfun preference from env, then expyfun config

    Parameters
    ----------
    key : str
        The preference key to look for. The os environment is searched first,
        then the expyfun config file is parsed.
    default : str | None
        Value to return if the key is not found.
    raise_error : bool
        If True, raise an error if the key is not found (instead of returning
        default).

    Returns
    -------
    value : str | None
        The preference key value.
    """
    if key is not None and not isinstance(key, string_types):
        raise ValueError('key must be a string')

    # first, check to see if key is in env
    if key is not None and key in os.environ:
        return os.environ[key]

    # second, look for it in expyfun config file
    config_path = get_config_path()
    if not op.isfile(config_path):
        key_found = False
        val = default
    else:
        with open(config_path, 'r') as fid:
            config = json.load(fid)
        if key is None:
            return config
        key_found = True if key in config else False
        val = config.get(key, default)

    if not key_found and raise_error is True:
        meth_1 = 'os.environ["%s"] = VALUE' % key
        meth_2 = 'expyfun.utils.set_config("%s", VALUE)' % key
        raise KeyError('Key "%s" not found in environment or in the '
                       'expyfun config file:\n%s\nTry either:\n'
                       '    %s\nfor a temporary solution, or:\n'
                       '    %s\nfor a permanent one. You can also '
                       'set the environment variable before '
                       'running python.'
                       % (key, config_path, meth_1, meth_2))
    return val


def set_config(key, value):
    """Set expyfun preference in config

    Parameters
    ----------
    key : str | None
        The preference key to set. If None, a tuple of the valid
        keys is returned, and ``value`` is ignored.
    value : str |  None
        The value to assign to the preference key. If None, the key is
        deleted.
    """
    if key is None:
        return sorted(known_config_types)
    if not isinstance(key, string_types):
        raise ValueError('key must be a string')
    # While JSON allow non-string types, we allow users to override config
    # settings using env, which are strings, so we enforce that here
    if not isinstance(value, string_types) and value is not None:
        raise ValueError('value must be a string or None')
    if key not in known_config_types and not \
            any(k in key for k in known_config_wildcards):
        warnings.warn('Setting non-standard config type: "%s"' % key)

    # Read all previous values
    config_path = get_config_path()
    if op.isfile(config_path):
        with open(config_path, 'r') as fid:
            config = json.load(fid)
    else:
        config = dict()
        logger.info('Attempting to create new expyfun configuration '
                    'file:\n%s' % config_path)
    if value is None:
        config.pop(key, None)
    else:
        config[key] = value

    # Write all values
    directory = op.split(config_path)[0]
    if not op.isdir(directory):
        os.mkdir(directory)
    with open(config_path, 'w') as fid:
        json.dump(config, fid, sort_keys=True, indent=0)


###############################################################################
# MISC


def fake_button_press(ec, button='1', delay=0.):
    """Fake a button press after a delay

    Notes
    -----
    This function only works with the keyboard controller (not TDT)!
    It uses threads to ensure that control is passed back, so other commands
    can be called (like wait_for_presses).
    """
    def send():
        ec._response_handler._on_pyglet_keypress(button, [], True)
    Timer(delay, send).start() if delay > 0. else send()


def fake_mouse_click(ec, pos, button='left', delay=0.):
    """Fake a mouse click after a delay"""
    button = dict(left=1, middle=2, right=4)[button]  # trans to pyglet

    def send():
        ec._mouse_handler._on_pyglet_mouse_click(pos[0], pos[1], button, [])
    Timer(delay, send).start() if delay > 0. else send()


def _check_pyglet_version(raise_error=False):
    """Check pyglet version, return True if usable.
    """
    import pyglet
    is_usable = LooseVersion(pyglet.version) >= LooseVersion('1.2')
    if raise_error is True and is_usable is False:
        raise ImportError('On Linux, you must run at least Pyglet '
                          'version 1.2, and you are running '
                          '{0}'.format(pyglet.version))
    return is_usable


def _wait_secs(secs, ec=None):
    """Wait a specified number of seconds.

    Parameters
    ----------
    secs : float
        Number of seconds to wait.
    ec : None | expyfun.ExperimentController instance
        The ExperimentController.

    Notes
    -----
    This function uses a while loop. Although this slams the CPU, it will
    guarantee that events (keypresses, etc.) are processed.
    """
    # hog the cpu, checking time
    t0 = clock()
    if ec is not None:
        while (clock() - t0) < secs:
            ec._dispatch_events()
            ec.check_force_quit()
            time.sleep(0.0001)
    else:
        wins = _get_display().get_windows()
        while (clock() - t0) < secs:
            for win in wins:
                win.dispatch_events()
            time.sleep(0.0001)


def running_rms(signal, win_length):
    """RMS of ``signal`` with rectangular window ``win_length`` samples long.

    Parameters
    ----------
    signal : array_like
        The (1-dimesional) signal of interest.
    win_length : int
        Length (in samples) of the rectangular window
    """
    assert signal.ndim == 1
    assert win_length > 0
    # The following is equivalent to:
    # sqrt(convolve(signal ** 2, ones(win_length) / win_length, 'valid'))
    # But an order of magnitude faster: 60 ms vs 7 ms for:
    #
    #     x = np.random.RandomState(0).randn(1000001)
    #     %timeit expyfun._utils.running_rms(x, 441)
    #
    sig2 = signal * signal
    c1 = np.cumsum(sig2)
    out = c1[win_length - 1:].copy()
    if len(out) == 0:  # len(signal) < len(win_length)
        out = np.array([np.sqrt(c1[-1] / signal.size)])
    else:
        out[1:] -= c1[:-win_length]
        out /= win_length
        np.sqrt(out, out=out)
    return out


def _fix_audio_dims(signal, n_channels):
    """Make it so a valid audio buffer is in the standard dimensions.

    Parameters
    ----------
    signal : array_like
        The signal whose dimensions should be checked and fixed.
    n_channels : int
        The number of channels that the output should have.
        If the input is mono and n_channels=2, it will be tiled to be
        shape (2, n_samples). Otherwise, the number of channels in signal
        must match n_channels.

    Returns
    -------
    signal_fixed : array
        The signal with standard dimensions (n_channels, N).
    """
    # Check requested channel output
    n_channels = int(operator.index(n_channels))
    signal = np.asarray(np.atleast_2d(signal), dtype=np.float32)
    # Check dimensionality
    if signal.ndim != 2:
        raise ValueError('Sound data must have one or two dimensions, got %s.'
                         % (signal.ndim,))
    # Return data with correct dimensions
    if n_channels == 2 and signal.shape[0] == 1:
        signal = np.tile(signal, (n_channels, 1))
    if signal.shape[0] != n_channels:
        raise ValueError('signal channel count %d did not match required '
                         'channel count %d' % (signal.shape[0], n_channels))
    return signal


def _sanitize(text_like):
    """Cast as string, encode as UTF-8 and sanitize any escape characters.
    """
    return text_type(text_like).encode('unicode_escape').decode('utf-8')


def _sort_keys(x):
    """Sort and return keys of dict"""
    keys = list(x.keys())  # note: not thread-safe
    idx = np.argsort([str(k) for k in keys])
    keys = [keys[ii] for ii in idx]
    return keys


def object_diff(a, b, pre=''):
    """Compute all differences between two python variables

    Parameters
    ----------
    a : object
        Currently supported: dict, list, tuple, ndarray, int, str, bytes,
        float, StringIO, BytesIO.
    b : object
        Must be same type as ``a``.
    pre : str
        String to prepend to each line.

    Returns
    -------
    diffs : str
        A string representation of the differences.

    Notes
    -----
    Taken from mne-python with permission.
    """
    out = ''
    if type(a) != type(b):
        out += pre + ' type mismatch (%s, %s)\n' % (type(a), type(b))
    elif isinstance(a, dict):
        k1s = _sort_keys(a)
        k2s = _sort_keys(b)
        m1 = set(k2s) - set(k1s)
        if len(m1):
            out += pre + ' x1 missing keys %s\n' % (m1)
        for key in k1s:
            if key not in k2s:
                out += pre + ' x2 missing key %s\n' % key
            else:
                out += object_diff(a[key], b[key], pre + 'd1[%s]' % repr(key))
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            out += pre + ' length mismatch (%s, %s)\n' % (len(a), len(b))
        else:
            for xx1, xx2 in zip(a, b):
                out += object_diff(xx1, xx2, pre='')
    elif isinstance(a, (string_types, int, float, bytes)):
        if a != b:
            out += pre + ' value mismatch (%s, %s)\n' % (a, b)
    elif a is None:
        if b is not None:
            out += pre + ' a is None, b is not (%s)\n' % (b)
    elif isinstance(a, np.ndarray):
        if not np.array_equal(a, b):
            out += pre + ' array mismatch\n'
    else:
        raise RuntimeError(pre + ': unsupported type %s (%s)' % (type(a), a))
    return out


def _check_skip_backend(backend):
    from expyfun._sound_controllers import _import_backend
    import pytest
    if isinstance(backend, dict):  # actually an AC
        backend = backend['SOUND_CARD_BACKEND']
    try:
        _import_backend(backend)
    except Exception as exc:
        pytest.skip('Skipping test for backend %s: %s' % (backend, exc))


def _check_params(params, keys, defaults, name):
    if not isinstance(params, dict):
        raise TypeError('{0} must be a dict, got type {1}'
                        .format(name, type(params)))
    params = deepcopy(params)
    if not isinstance(params, dict):
        raise TypeError('{0} must be a dict, got {1}'
                        .format(name, type(params)))
    # Set sensible defaults for values that are not passed
    for k in keys:
        params[k] = params.get(k, get_config(k, defaults.get(k, None)))
    # Check keys
    for k in params.keys():
        if k not in keys:
            raise KeyError('Unrecognized key in {0}["{1}"], must be '
                           'one of {2}'.format(name, k, ', '.join(keys)))
    return params


def _get_display():
    import pyglet
    try:
        display = pyglet.canvas.get_display()
    except AttributeError:  # < 1.4
        display = pyglet.window.get_platform().get_default_display()
    return display
