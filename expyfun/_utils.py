"""Some utility functions"""

# Authors: Eric Larson <larsoner@uw.edu>
#
# License: BSD (3-clause)

import warnings
import scipy as sp
import os
import os.path as op
import inspect
import sys
import tempfile
from shutil import rmtree
import atexit
import json
from distutils.version import LooseVersion
import pyglet
from numpy import sqrt, convolve, ones
from numpy.testing.decorators import skipif
import logging
import datetime
from timeit import default_timer as clock
from ._externals import decorator

# set this first thing to make sure it "takes"
pyglet.options['debug_gl'] = False

try:
    import pylink  # noqa, analysis:ignore
except ImportError:
    has_pylink = False
else:
    has_pylink = True

try:
    import pandas  # noqa, analysis:ignore
except ImportError:
    has_pandas = False
else:
    has_pandas = True


# for py3k (eventually)
try:
    string_types = basestring  # noqa
except NameError:
    string_types = str  # noqa
try:
    input = raw_input  # input is raw_input in py3k
except NameError:
    input = input

try:
    x = unicode('test')  # noqa
except NameError:
    unicode = str

try:
    from urllib2 import urlopen
except ImportError:
    from urllib.request import urlopen

try:
    import tables  # noqa, analysis:ignore
except Exception:
    has_pytables = False
else:
    has_pytables = True


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
        if verbose is True:
            verbose = 'INFO'
        else:
            verbose = 'WARNING'
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
                 output_format='%(asctime)s - %(levelname)-5s - %(message)s',
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


def _check_units(units):
    """Helper to make sure user passed in valid units type"""
    good_units = ['norm', 'pix', 'deg']
    if units not in good_units:
        raise ValueError('"units" must be one of {}, not {}'
                         ''.format(good_units, units))


def _check_pytables():
    """Helper to error if Pytables is not found"""
    if not has_pytables:
        raise ImportError('pytables could not be imported')
    import tables as tb
    return tb


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
    arg_names = inspect.getargspec(function).args

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


requires_pylink = skipif(has_pylink is False, 'Requires functional pylink')
requires_pandas = skipif(has_pandas is False, 'Requires pandas')
requires_pytables = skipif(has_pytables is False, 'Requires pytables')


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
    fname_url = 'https://lester.ilabs.uw.edu/files/{0}'.format(fname)
    if not op.isfile(fname_out):
        try:
            with open(fname_out, 'wb') as fid:
                www = urlopen(fname_url, timeout=3.0)
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
known_config_types = ['RESPONSE_DEVICE',
                      'AUDIO_CONTROLLER',
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
                      'EXPYFUN_INTERACTIVE_TESTING',
                      'EXPYFUN_LOGGING_LEVEL',
                      ]

# These allow for partial matches: 'NAME_1' is okay key if 'NAME' is listed
known_config_wildcards = []


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

    if not isinstance(key, string_types):
        raise ValueError('key must be a string')

    # first, check to see if key is in env
    if key in os.environ:
        return os.environ[key]

    # second, look for it in expyfun config file
    config_path = get_config_path()
    if not op.isfile(config_path):
        key_found = False
        val = default
    else:
        with open(config_path, 'r') as fid:
            config = json.load(fid)
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
    key : str
        The preference key to set.
    value : str |  None
        The value to assign to the preference key. If None, the key is
        deleted.
    """

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

def _check_pyglet_version(raise_error=False):
    """Check pyglet version, return True if usable.
    """
    is_usable = LooseVersion(pyglet.version) >= LooseVersion('1.2')
    if raise_error is True and is_usable is False:
        raise ImportError('On Linux, you must run at least Pyglet '
                          'version 1.2, and you are running '
                          '{0}'.format(pyglet.version))
    return is_usable


interactive_test = skipif(get_config('EXPYFUN_INTERACTIVE_TESTING', 'False') !=
                          'True', 'Interactive testing disabled.')


def wait_secs(secs, ec=None):
    """Wait a specified number of seconds.

    Parameters
    ----------
    secs : float
        Number of seconds to wait.

    Notes
    -----
    This function uses a while loop. Although this slams the CPU, it will
    guarantee that events (keypresses, etc.) are processed.
    """
    #hog the cpu, checking time
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


def _sanitize(text_like):
    """Cast as string, encode as UTF-8 and sanitize any escape characters.
    """
    return unicode(text_like).encode('unicode_escape').decode('utf-8')
