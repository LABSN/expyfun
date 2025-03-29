import os
import sys
import warnings
from importlib import reload
from io import StringIO
from os import path as op
from pathlib import Path

from ._utils import _TempDir, run_subprocess
from ._version import __version__

this_version = __version__[-7:]

try:
    run_subprocess(["git", "--help"])
except Exception as exp:
    _has_git, why_not = False, str(exp)
else:
    _has_git, why_not = True, None


def _check_git():
    """Helper to check the expyfun version"""
    if not _has_git:
        raise RuntimeError(f"git not found: {why_not}")


def _check_version_format(version):
    """Helper to ensure version is of proper format"""
    if not isinstance(version, str) or len(version) != 7:
        raise TypeError(f"version must be a string of length 7, got {version}")


def _active_version(wd):
    """Helper to get the currently active version"""
    return run_subprocess(["git", "rev-parse", "HEAD"], cwd=wd)[0][:7]


def download_version(version="current", dest_dir=None):
    """Download specific expyfun version

    Parameters
    ----------
    version : str
        Version to check out (7-character git commit number).
        Can also be ``'current'`` (default) to download whatever the
        latest ``upstream/master`` version is.
    dest_dir : str | None
        Destination directory. If None, the current working
        directory is used.

    Notes
    -----
    This function requires installation of ``gitpython``.
    """
    _check_git()
    _check_version_format(version)
    if dest_dir is None:
        dest_dir = os.getcwd()
    if not isinstance(dest_dir, str) or not op.isdir(dest_dir):
        raise OSError(f"Destination directory {dest_dir} does not exist")
    if op.isdir(op.join(dest_dir, "expyfun")):
        raise OSError(
            f'Destination directory {dest_dir} already has "expyfun" subdirectory'
        )

    # fetch locally and get the proper version
    tempdir = _TempDir()
    expyfun_dir = op.join(tempdir, "expyfun")  # git will auto-create this dir
    repo_url = "https://github.com/LABSN/expyfun.git"
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"  # do not prompt for credentials
    run_subprocess(
        ["git", "clone", repo_url, expyfun_dir, "--single-branch", "--branch", "main"],
        env=env,
    )
    version = _active_version(expyfun_dir) if version == "current" else version
    try:
        run_subprocess(["git", "checkout", version], cwd=expyfun_dir, env=env)
    except Exception as exp:
        raise RuntimeError(f"Could not check out version {version}: {str(exp)}")
    assert _active_version(expyfun_dir) == version

    # install
    orig_dir = os.getcwd()
    os.chdir(expyfun_dir)
    # monkey-patch setup.cfg if present to avoid newer setuptools errors
    setup_cfg = Path(expyfun_dir) / "setup.cfg"
    if setup_cfg.is_file():
        setup_cfg.write_text(
            setup_cfg.read_text("utf-8").replace(
                "\ndoc-files = doc",
                "\ndoc_files = doc",
            )
        )
    setup_py = Path(expyfun_dir) / "setup.py"
    if setup_py.is_file():  # should always be true but okay
        # Ensure setup.py is compatible with the current setuptools
        # This is a workaround for some issues with setuptools in newer versions
        text = setup_py.read_text("utf-8")
        text = text.replace(
            "from numpy.distutils.core import setup",
            "from setuptools import setup",
        )
        text = "\n".join(
            line for line in text.splitlines() if "License :: OSI Approved" not in line
        )
        setup_py.write_text(text)

    # ensure our version-specific "setup" is imported
    sys.path.insert(0, expyfun_dir)
    orig_stdout = sys.stdout
    try:
        # on pytest with Py3k this can be problematic
        if "setup" in sys.modules:
            del sys.modules["setup"]
        import setup

        reload(setup)
        setup_version = setup.git_version()
        # This is necessary because for a while git_version returned
        # a tuple of (version, fork)
        if not isinstance(setup_version, str):
            setup_version = setup_version[0]
        assert version.lower() == setup_version[:7].lower()
        del setup_version
        # Now we need to monkey-patch to change FULL_VERSION, which can be for example:
        # 2.0.0.dev-090948e
        # to
        # 2.0.0.dev0+090948e
        if "-" in setup.FULL_VERSION:
            setup.FULL_VERSION = setup.FULL_VERSION.replace("-", "0+")  # PEP440
        sys.stdout = StringIO()
        with warnings.catch_warnings(record=True):  # PEP440
            setup.setup_package(script_args=["build", "--build-purelib", dest_dir])
    finally:
        sys.stdout = orig_stdout
        sys.path.pop(sys.path.index(expyfun_dir))
        os.chdir(orig_dir)
    print(
        "\n".join(
            [
                "Successfully checked out expyfun version:",
                version,
                "into destination directory:",
                op.join(dest_dir),
            ]
        )
    )


def assert_version(version):
    """Assert that a specific version of expyfun is imported

    Parameters
    ----------
    version : str
        Version to check (7 characters).
    """
    _check_version_format(version)
    if this_version.lower() != version.lower():
        raise AssertionError(
            f"Requested version {version} does not match current version {this_version}"
        )
