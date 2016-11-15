"""Some utility functions"""

# Authors: Eric Larson <larsoner@uw.edu>
#
# License: BSD (3-clause)

import warnings
import subprocess
import numpy as np
import scipy as sp
import os
import os.path as op
import inspect
import sys
import tempfile
import ssl
from shutil import rmtree
import atexit
import json
from functools import partial
from distutils.version import LooseVersion
from numpy import sqrt, convolve, ones
from numpy.testing.decorators import skipif
import logging
import datetime
from timeit import default_timer as clock
from threading import Timer

from ._externals import decorator

# set this first thing to make sure it "takes"
try:
    import pyglet
    pyglet.options['debug_gl'] = False
    del pyglet
except Exception:
    pass

try:
    import pandas  # noqa, analysis:ignore
except ImportError:
    has_pandas = False
else:
    has_pandas = True

try:
    import h5py  # noqa, analysis:ignore
except Exception:
    has_h5py = False
else:
    has_h5py = True

try:
    import joblib  # noqa, analysis:ignore
except Exception:
    has_joblib = False
else:
    has_joblib = True

# for py3k (eventually)
if sys.version.startswith('2'):
    string_types = basestring  # noqa
    input = raw_input  # noqa, input is raw_input in py3k
    text_type = unicode  # noqa
    from urllib2 import urlopen  # noqa
    from cStringIO import StringIO  # noqa
else:
    string_types = str
    text_type = str
    from urllib.request import urlopen
    input = input
    from io import StringIO  # noqa, analysis:ignore

###############################################################################
# LOGGING

EXP = 25
logging.addLevelName(EXP, 'EXP')


def exp(self, message, *args, **kwargs):
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
    """Clock that uses "clock" function but starts at zero on init"""
    def __init__(self):
        self._start_time = clock()

    def get_time(self):
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
    """Ridiculous class to work around how doctest captures stdout"""
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
        Must be ``'norm'``, ``'deg'``, or ``'pix'``.
    """
    good_units = ['norm', 'pix', 'deg']
    if units not in good_units:
        raise ValueError('"units" must be one of {}, not {}'
                         ''.format(good_units, units))


###############################################################################
# DECORATORS

# Following deprecated class copied from scikit-learn

# force show of DeprecationWarning even on python 2.7
warnings.simplefilter('default')


class deprecated(object):
    """Decorator to mark a function or class as deprecated.

    Issue a warning when the function is called/the class is instantiated and
    adds a warning to the docstring.

    The optional extra argument will be appended to the deprecation message
    and the docstring. Note: to use this with the default value for extra, put
    in an empty of parentheses:

    >>> from expyfun.utils import deprecated
    >>> deprecated() # doctest: +ELLIPSIS
    <expyfun.utils.deprecated object at ...>

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
    function - function
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
        except:
            set_log_level(old_level)
            raise
        set_log_level(old_level)
        return ret
    else:
        ret = function(*args, **kwargs)
        return ret


def requires_avbin():
    try:
        from pyglet.media.avbin import AVbinSource
        del AVbinSource
        _has_avbin = True
    except ImportError:
        _has_avbin = False
    return skipif(not _has_avbin, 'Requires AVbin')


_is_appveyor = (os.getenv('APPVEYOR', 'False').lower() == 'true')
requires_pandas = skipif(has_pandas is False, 'Requires pandas')
requires_h5py = skipif(has_h5py is False, 'Requires h5py')
requires_joblib = skipif(has_joblib is False, 'Requires joblib')
requires_opengl21 = skipif(_is_appveyor, 'Appveyor OpenGL too old')


def _has_scipy_version(version):
    return (LooseVersion(sp.__version__) >= LooseVersion(version))


def _hide_window(function):
    """Decorator to hide expyfun windows during testing"""
    import nose

    def dec(*args, **kwargs):
        orig_val = os.getenv('_EXPYFUN_WIN_INVISIBLE')
        try:
            os.environ['_EXPYFUN_WIN_INVISIBLE'] = 'true'
            out = function(*args, **kwargs)
            return out
        finally:
            if orig_val is None:
                del os.environ['_EXPYFUN_WIN_INVISIBLE']
            else:
                os.environ['_EXPYFUN_WIN_INVISIBLE'] = orig_val
    return nose.tools.make_decorator(function)(dec)


###############################################################################
# CONFIG / PREFS


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
                www = this_urlopen(fname_url, timeout=10.0)
                fid.write(www.read())
                www.close()
        except Exception:
            os.remove(fname_out)
            raise
    return fname_out


def get_config_path():
    """Get path to standard expyfun config file

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
                      'TDT_MODEL',
                      'TDT_INTERFACE',
                      'TDT_CIRCUIT_PATH',
                      'TRIGGER_CONTROLLER',
                      'WINDOW_SIZE',
                      'SCREEN_NUM',
                      'SCREEN_WIDTH',
                      'SCREEN_DISTANCE',
                      'SCREEN_SIZE_PIX',
                      'EXPYFUN_LOGGING_LEVEL',
                      )

# These allow for partial matches: 'NAME_1' is okay key if 'NAME' is listed
known_config_wildcards = ()


def get_config(key, default=None, raise_error=False):
    """Read expyfun preference from env, then expyfun config

    Parameters
    ----------
    key : str
        The preference key to look for. The os evironment is searched first,
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


def wait_secs(secs, ec=None):
    """Wait a specified number of seconds.

    Parameters
    ----------
    secs : float
        Number of seconds to wait.
    ec : None | expyfun.ExperimentController instance

    Notes
    -----
    This function uses a while loop. Although this slams the CPU, it will
    guarantee that events (keypresses, etc.) are processed.
    """
    # hog the cpu, checking time
    import pyglet
    t0 = clock()
    wins = pyglet.window.get_platform().get_default_display().get_windows()
    while (clock() - t0) < secs:
        for win in wins:
            win.dispatch_events()
        if ec is not None:
            ec.check_force_quit()


def running_rms(signal, win_length):
    """RMS of ``signal`` with rectangular window ``win_length`` samples long.

    Parameters
    ----------
    signal : array_like
        The (1-dimesional) signal of interest.
    win_length : int
        Length (in samples) of the rectangular window
    """
    return sqrt(convolve(signal ** 2, ones(win_length) / win_length, 'valid'))


def _fix_audio_dims(signal, n_channels=None):
    """Make it so a valid audio buffer is in the standard dimensions

    Parameters
    ----------
    signal : array_like
        The signal whose dimensions should be checked and fixed.
    n_channels : int or None
        The number of channels that the output should have. If ``None``, don't
        change the number of channels (and assume vectors have one channel).
        Setting ``n_channels`` to 1 when the input is stereo will result in an
        error, since stereo-mono conversion is non-trivial and beyond the
        scope of this function.

    Returns
    -------
    signal_fixed : array
        The signal with standard dimensions (1, N) or (2, N).
    """
    # Check requested channel output
    if n_channels not in (None, 1, 2):
        raise ValueError('Number of channels out must be None, 1, or 2.')

    signal = np.asarray(signal, dtype=np.float32)

    # Check dimensionality
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]
    elif signal.ndim == 2:
        if np.min(signal.shape) > 2:
            raise ValueError('Sound data has more than two channels.')
        if signal.shape[0] > 2:  # Needs to be correct for remainder of checks
            signal = signal.T
        if signal.shape[0] not in [1, 2]:
            raise ValueError('Audio shape must be (N,), (1, N), or (2, N).')
        if signal.shape[0] == 2 and n_channels == 1:
            raise ValueError('Requested mono output but gave stereo input.')
    else:
        raise ValueError('Sound data must have one or two dimensions.')
    # Return data with correct dimensions
    if n_channels is not None and n_channels != signal.shape[0]:
        signal = np.tile(signal, (n_channels, 1))
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
