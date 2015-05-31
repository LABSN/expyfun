import expyfun
import fnmatch
import os
from os import path as op


def test_ascii():
    """Test that all files only have ASCII characters (for docs)"""
    src = op.abspath(op.dirname(expyfun.__file__))
    if op.isdir(op.join(src, '..', '..', 'expyfun')):
        src = op.abspath(op.join(src, '..', 'doc'))
    assert op.isdir(src)
    matches = []
    types = ('*.py', '*.rst', '*.txt', '*.cfg')
    for root, dirnames, filenames in os.walk(src):
        for ty in types:
            for filename in fnmatch.filter(filenames, ty):
                matches.append(os.path.join(root, filename))
    for fname in matches:
        with open(fname, 'rb') as fid:
            for li, line in enumerate(fid.readlines()):
                try:
                    line.decode('ascii')
                except UnicodeDecodeError:
                    raise ValueError('file {0} has non-ascii char on line {1}:'
                                     '\n{2}'.format(fname, li, line))
