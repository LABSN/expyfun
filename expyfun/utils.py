"""Some utility functions"""

# Authors: Eric Larson <larsoner@uw.edu>
#
# License: BSD (3-clause)

import warnings
from psychopy import logging as psylog
import os
import os.path as op
from functools import wraps
import inspect
import sys
import tempfile
from shutil import rmtree
import atexit
import json
from distutils.version import LooseVersion
import pyglet
import platform
from numpy.testing.decorators import skipif
from psychopy import core
try:
    import pylink
except ImportError:
    has_pylink = False
else:
    has_pylink = True


###############################################################################
# RANDOM UTILITIES

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
                print 'Deleting %s ...' % self._path
            rmtree(self._path, ignore_errors=True)


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


def verbose_dec(function):
    """Decorator to allow functions to override default log level

    Do not call this function directly to set the global verbosity level,
    instead use set_log_level().

    Parameters (to decorated function)
    ----------------------------------
    verbose : bool, str, int, or None
        The level of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL. Note that these are for
        convenience and are equivalent to passing in logging.DEBUG, etc.
        For bool, True is the same as 'INFO', False is the same as 'WARNING'.
        None defaults to using the current log level [e.g., set using
        expyfun.set_log_level()].
    """
    arg_names = inspect.getargspec(function).args
    # this wrap allows decorated functions to be pickled (e.g., for parallel)

    @wraps(function)
    def dec(*args, **kwargs):
        # Check if the first arg is "self", if it has verbose, make it default
        if len(arg_names) > 0 and arg_names[0] == 'self':
            default_level = getattr(args[0], 'verbose', None)
        else:
            default_level = None
        verbose_level = kwargs.get('verbose', default_level)
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
            return function(*args, **kwargs)

    # set __wrapped__ attribute so ?? in IPython gets the right source
    dec.__wrapped__ = function

    return dec


requires_pylink = skipif(has_pylink is False, 'Requires functional pylink')

###############################################################################
# LOGGING

def set_log_level(verbose=None, return_old_level=False):
    """Convenience function for setting the logging level

    Parameters
    ----------
    verbose : bool, str, int, or None
        The verbosity of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL. Note that these are for
        convenience and are equivalent to passing in logging.DEBUG, etc.
        For bool, True is the same as 'INFO', False is the same as 'WARNING'.
        If None, the environment variable EXPYFUN_LOG_LEVEL is read, and if
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
    if isinstance(verbose, basestring):
        verbose = verbose.upper()
        logging_types = dict(DEBUG=psylog.DEBUG, INFO=psylog.INFO,
                             WARNING=psylog.WARNING, ERROR=psylog.ERROR,
                             CRITICAL=psylog.CRITICAL)
        if not verbose in logging_types:
            raise ValueError('verbose must be of a valid type')
        verbose = logging_types[verbose]

    logger = psylog.console
    old_verbose = logger.level
    logger.setLevel(verbose)
    return (old_verbose if return_old_level else None)


###############################################################################
# CONFIG / PREFS

def get_config_path():
    """Get path to standard expyfun config file

    Returns
    -------
    config_path : str
        The path to the expyfun configuration file. On windows, this
        will be '%APPDATA%\.expyfun\expyfun.json'. On every other
        system, this will be $HOME/.expyfun/expyfun.json.
    """

    # this has been checked on OSX64, Linux64, and Win32
    val = os.getenv('APPDATA' if 'nt' == os.name.lower() else 'HOME', None)
    if val is None:
        raise ValueError('expyfun config file path could '
                         'not be determined, please report this '
                         'error to expyfun developers')

    val = op.join(val, '.expyfun', 'expyfun.json')
    return val


# List the known configuration values
known_config_types = ['RESPONSE_DEVICE',
                      'AUDIO_CONTROLLER',
                      'TDT_MODEL',
                      'TDT_INTERFACE',
                      'TDT_CIRCUIT_PATH',
                      'WINDOW_SIZE',
                      'SCREEN_NUM',
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

    if not isinstance(key, basestring):
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

    if not isinstance(key, basestring):
        raise ValueError('key must be a string')
    # While JSON allow non-string types, we allow users to override config
    # settings using env, which are strings, so we enforce that here
    if not isinstance(value, basestring) and value is not None:
        raise ValueError('value must be a string or None')
    if not key in known_config_types and not \
            any(k in key for k in known_config_wildcards):
        warnings.warn('Setting non-standard config type: "%s"' % key)

    # Read all previous values
    config_path = get_config_path()
    if op.isfile(config_path):
        with open(config_path, 'r') as fid:
            config = json.load(fid)
    else:
        config = dict()
        psylog.info('Attempting to create new expyfun configuration '
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


def _check_pyglet_version(raise_error=False):
    """Check pyglet version, return True if usable.
    """
    is_usable = (LooseVersion(pyglet.version) >= LooseVersion('1.2')
                 or platform.system() != 'Linux')
    if raise_error is True and is_usable is False:
        raise ImportError('On Linux, you must run at least Pyglet '
                          'version 1.2, and you are running '
                          '{0}'.format(pyglet.version))
    return is_usable


interactive_test = skipif(get_config('EXPYFUN_INTERACTIVE_TESTING', 'False') !=
                          'True', 'Interactive testing disabled.')

tdt_test = skipif(get_config('AUDIO_CONTROLLER', 'psychopy') != 'tdt',
                  'TDT not set in system config.')


def wait_secs(secs, hog_cpu_time=0.2):
    """Wait a specified number of seconds.

    Parameters
    ----------
    secs : float
        Number of seconds to wait.
    hog_cpu_time : float
        Amount of CPU time to hog. See Notes.

    Notes
    -----
    From the PsychoPy documentation:
    If secs=10 and hogCPU=0.2 then for 9.8s python's time.sleep function
    will be used, which is not especially precise, but allows the cpu to
    perform housekeeping. In the final hogCPUperiod the more precise method
    of constantly polling the clock is used for greater precision.

    If you want to obtain key-presses during the wait, be sure to use
    pyglet and to hogCPU for the entire time, and then call
    psychopy.event.getKeys() after calling wait().

    If you want to suppress checking for pyglet events during the wait, do
    this once:
        core.checkPygletDuringWait = False
    and from then on you can do:
        core.wait(sec)
    This will preserve terminal-window focus during command line usage.
    """
    if any([secs < 0.2, secs < hog_cpu_time]):
        hog_cpu_time = secs
    core.wait(secs, hogCPUperiod=hog_cpu_time)
