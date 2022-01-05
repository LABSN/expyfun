import inspect
from inspect import getsource
import os.path as op
from pkgutil import walk_packages
import re
import sys
from unittest import SkipTest
import warnings

import pytest

import expyfun
from expyfun._utils import _get_args

public_modules = [
    # the list of modules users need to access for all functionality
    'expyfun',
    'expyfun.stimuli',
    'expyfun.io',
    'expyfun.visual',
    'expyfun.codeblocks',
    'expyfun.analyze',
]


def requires_numpydoc(fun):
    """Decorate based on numpydoc existence."""
    try:
        from numpydoc import docscrape  # noqa
    except ImportError:
        have = False
    else:
        have = True
    return pytest.mark.skipif(not have, reason='Requires numpydoc')(fun)


def get_name(func, cls=None):
    """Get the name."""
    parts = []
    module = inspect.getmodule(func)
    if module:
        parts.append(module.__name__)
    if cls is not None:
        parts.append(cls.__name__)
    parts.append(func.__name__)
    return '.'.join(parts)


# functions to ignore args / docstring of
docstring_ignores = [
]
char_limit = 800  # XX eventually we should probably get this lower
docstring_length_ignores = [
]
tab_ignores = [
]
_doc_special_members = []


def check_parameters_match(func, doc=None, cls=None):
    """Check docstring, return list of incorrect results."""
    from numpydoc import docscrape
    incorrect = []
    name_ = get_name(func, cls=cls)
    if not name_.startswith('expyfun.') or \
            name_.startswith('expyfun._externals'):
        return incorrect
    if inspect.isdatadescriptor(func):
        return incorrect
    args = _get_args(func)
    # drop self
    if len(args) > 0 and args[0] == 'self':
        args = args[1:]

    if doc is None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            try:
                doc = docscrape.FunctionDoc(func)
            except Exception as exp:
                incorrect += [name_ + ' parsing error: ' + str(exp)]
                return incorrect
        if len(w):
            raise RuntimeError('Error for %s:\n%s' % (name_, w[0]))
    # check set
    parameters = doc['Parameters']
    # clean up some docscrape output:
    parameters = [[p[0].split(':')[0].strip('` '), p[2]]
                  for p in parameters]
    parameters = [p for p in parameters if '*' not in p[0]]
    param_names = [p[0] for p in parameters]
    if len(param_names) != len(args):
        bad = str(sorted(list(set(param_names) - set(args)) +
                         list(set(args) - set(param_names))))
        if not any(re.match(d, name_) for d in docstring_ignores) and \
                'deprecation_wrapped' not in func.__code__.co_name:
            incorrect += [name_ + ' arg mismatch: ' + bad]
    else:
        for n1, n2 in zip(param_names, args):
            if n1 != n2:
                incorrect += [name_ + ' ' + n1 + ' != ' + n2]
        for param_name, desc in parameters:
            desc = '\n'.join(desc)
            full_name = name_ + '::' + param_name
            if full_name in docstring_length_ignores:
                assert len(desc) > char_limit  # assert it actually needs to be
            elif len(desc) > char_limit:
                incorrect += ['%s too long (%d > %d chars)'
                              % (full_name, len(desc), char_limit)]
    return incorrect


@requires_numpydoc
def test_docstring_parameters():
    """Test module docstring formatting."""
    from numpydoc import docscrape
    incorrect = []
    for name in public_modules:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('ignore')
            module = __import__(name, globals())
        for submod in name.split('.')[1:]:
            module = getattr(module, submod)
        classes = inspect.getmembers(module, inspect.isclass)
        for cname, cls in classes:
            if cname.startswith('_') and cname not in _doc_special_members:
                continue
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                cdoc = docscrape.ClassDoc(cls)
            for ww in w:
                if 'Using or importing the ABCs' not in str(ww.message):
                    raise RuntimeError('Error for __init__ of %s in %s:\n%s'
                                       % (cls, name, ww))
            if hasattr(cls, '__init__'):
                incorrect += check_parameters_match(cls.__init__, cdoc, cls)
            for method_name in cdoc.methods:
                method = getattr(cls, method_name)
                incorrect += check_parameters_match(method, cls=cls)
            if hasattr(cls, '__call__'):
                incorrect += check_parameters_match(cls.__call__, cls=cls)
        functions = inspect.getmembers(module, inspect.isfunction)
        for fname, func in functions:
            if fname.startswith('_'):
                continue
            incorrect += check_parameters_match(func)
    msg = '\n' + '\n'.join(sorted(list(set(incorrect))))
    if len(incorrect) > 0:
        raise AssertionError(msg)


def test_tabs():
    """Test that there are no tabs in our source files."""
    # avoid importing modules that require mayavi if mayavi is not installed
    ignore = tab_ignores[:]
    for importer, modname, ispkg in walk_packages(expyfun.__path__,
                                                  prefix='expyfun.'):
        if not ispkg and modname not in ignore:
            # mod = importlib.import_module(modname)  # not py26 compatible!
            try:
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter('ignore')
                    __import__(modname)
            except Exception:  # can't import properly
                continue
            mod = sys.modules[modname]
            try:
                source = getsource(mod)
            except IOError:  # user probably should have run "make clean"
                continue
            assert '\t' not in source, ('"%s" has tabs, please remove them '
                                        'or add it to the ignore list'
                                        % modname)


documented_ignored_mods = (
)
documented_ignored_names = """
add_pad
fetch_data_file
get_band_freqs
get_bands
get_carriers
get_config
get_config_path
get_env
run_subprocess
set_log_file
verbose
""".split('\n')


def test_documented():
    """Test that public functions and classes are documented."""
    # skip modules that require mayavi if mayavi is not installed
    public_modules_ = public_modules[:]
    doc_file = op.abspath(op.join(op.dirname(__file__), '..', '..', 'doc',
                                  'python_reference.rst'))
    if not op.isfile(doc_file):
        raise SkipTest('Documentation file not found: %s' % doc_file)
    known_names = list()
    with open(doc_file, 'rb') as fid:
        for line in fid:
            line = line.decode('utf-8')
            if not line.startswith('  '):  # at least two spaces
                continue
            line = line.split()
            if len(line) == 1 and line[0] != ':':
                known_names.append(line[0].split('.')[-1])
    known_names = set(known_names)

    missing = []
    for name in public_modules_:
        with warnings.catch_warnings(record=True):  # traits warnings
            warnings.simplefilter('ignore')
            module = __import__(name, globals())
        for submod in name.split('.')[1:]:
            module = getattr(module, submod)
        classes = inspect.getmembers(module, inspect.isclass)
        functions = inspect.getmembers(module, inspect.isfunction)
        checks = list(classes) + list(functions)
        for name, cf in checks:
            if not name.startswith('_') and name not in known_names:
                from_mod = inspect.getmodule(cf).__name__
                if (from_mod.startswith('expyfun') and
                        not from_mod.startswith('expyfun._externals') and
                        not any(from_mod.startswith(x)
                                for x in documented_ignored_mods) and
                        name not in documented_ignored_names):
                    missing.append('%s (%s.%s)' % (name, from_mod, name))
    if len(missing) > 0:
        raise AssertionError('\n\nFound new public members missing from '
                             'doc/python_reference.rst:\n\n* ' +
                             '\n* '.join(sorted(set(missing))))
