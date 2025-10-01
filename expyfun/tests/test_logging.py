import operator
import os

import pytest

from expyfun import ExperimentController
from expyfun._utils import _check_skip_backend, requires_lib

std_args = ["test"]
std_kwargs = dict(
    participant="foo",
    session="01",
    full_screen=False,
    window_size=(1, 1),
    verbose=True,
    noise_db=0,
    version="dev",
)


@requires_lib("mne")
def test_logging(ac, tmpdir, hide_window):
    """Test logging to file (Pyglet)."""
    if ac != "tdt":
        _check_skip_backend(ac)
    orig_dir = os.getcwd()
    os.chdir(str(tmpdir))
    kwargs = std_kwargs.copy()
    if isinstance(ac, dict) and ac.get("SOUND_CARD_BACKEND", "") == "sounddevice":
        kwargs["gapless"] = True
    try:
        with ExperimentController(
            *std_args,
            audio_controller=ac,
            response_device="keyboard",
            trigger_controller="dummy",
            **kwargs,
        ) as ec:
            test_name = ec._log_file
            stamp = ec.current_time
            ec.wait_until(stamp)  # wait_until called w/already passed timest.
            with pytest.warns(UserWarning, match="RMS"):
                ec.load_buffer([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])  # RMS warning

        with open(test_name) as fid:
            data = "\n".join(fid.readlines())

        # check for various expected log messages (TODO: add more)
        should_have = [
            "Participant: foo",
            "Session: 01",
            "wait_until was called",
            "Stimulus max RMS (",
        ]
        if ac == "tdt":
            should_have.append("TDT")
        else:
            should_have.append("sound card")
            if ac != "auto" and ac["SOUND_CARD_BACKEND"] != "auto":
                should_have.append(ac["SOUND_CARD_BACKEND"])
        assert_have_all(data, should_have)
    finally:
        os.chdir(orig_dir)


def assert_have_all(data, should_have):
    """Assert all substrings are in the logging output."""
    __tracebackhide__ = operator.methodcaller("errisinstance", AssertionError)
    for s in should_have:
        if s not in data:
            raise AssertionError(f'Missing data: "{s}" in:\n{data}')
