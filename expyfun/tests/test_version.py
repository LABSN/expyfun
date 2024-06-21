import os
import warnings
from os import path as op

import pytest

from expyfun import ExperimentController, __version__, assert_version, download_version
from expyfun._git import _has_git
from expyfun._utils import _TempDir


@pytest.mark.timeout(60)  # can be slow to download
def test_version_assertions():
    """Test version assertions."""
    pytest.raises(TypeError, assert_version, 1)
    pytest.raises(TypeError, assert_version, "1" * 8)
    pytest.raises(AssertionError, assert_version, "x" * 7)
    assert_version(__version__[-7:])

    # old, broken, new
    for wi, want_version in enumerate(("090948e", "cae6bc3", "b6e8a81")):
        print("Running %s" % want_version)
        tempdir = _TempDir()
        if not _has_git:
            pytest.raises(ImportError, download_version, want_version, tempdir)
        else:
            pytest.raises(
                IOError, download_version, want_version, op.join(tempdir, "foo")
            )
            pytest.raises(RuntimeError, download_version, "x" * 7, tempdir)
            ex_dir = op.join(tempdir, "expyfun")
            assert not op.isdir(ex_dir)
            with warnings.catch_warnings(record=True):  # Sometimes warns
                warnings.simplefilter("ignore")
                download_version(want_version, tempdir)
            assert op.isdir(ex_dir)
            assert op.isfile(op.join(ex_dir, "__init__.py"))
            got_fname = op.join(ex_dir, "_version.py")
            with open(got_fname) as fid:
                line1 = fid.readline().strip()
            got_version = line1.split(" = ")[1][-8:-1]
            ex = want_version
            if want_version == "cae6bc3":
                ex = (ex, ".dev0+c")
            assert got_version in ex, got_fname

            # auto dir determination
            orig_dir = os.getcwd()
            os.chdir(tempdir)
            try:
                assert op.isdir("expyfun")
                pytest.raises(IOError, download_version, want_version)
            finally:
                os.chdir(orig_dir)
    # make sure we can get latest version
    tempdir_2 = _TempDir()
    if _has_git:
        pytest.raises(IOError, download_version, dest_dir=tempdir)
        download_version(dest_dir=tempdir_2)


def test_integrated_version_checking():
    """Test EC version checking during init."""
    tempdir = _TempDir()
    args = ["test"]  # experiment name
    kwargs = dict(
        output_dir=tempdir,
        full_screen=False,
        window_size=(1, 1),
        participant="foo",
        session="01",
        stim_db=0.0,
        noise_db=0.0,
        verbose=True,
    )
    pytest.raises(RuntimeError, ExperimentController, *args, version=None, **kwargs)
    pytest.raises(
        AssertionError, ExperimentController, *args, version="59f3f5b", **kwargs
    )  # the very first commit
