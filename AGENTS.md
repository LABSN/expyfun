# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

## What this is

expyfun is a small library for high-precision auditory and visual stimulus delivery in
psychoacoustics / psychophysics experiments. The central object is `ExperimentController`, which
owns a pyglet window (drawn with raw OpenGL), an audio output (TDT hardware or a sound card), a
trigger controller, and keyboard/mouse/joystick input. Stimulus timing is the product: changes to
the audio or drawing paths must not add latency, jitter, or per-frame work.

## AI-assistance policy (read first)

- **Humans make every outward-facing interaction.** Agents do not open, comment on, review, or
  close issues or PRs, and do not commit or push unless the human explicitly asks for that
  specific action. Do not paste agent-written text into issues, PRs, or review comments; the human
  writes those in their own words.
- **Every change is reviewed by a human before it lands.** Leave changes in the working tree for
  the human to inspect with `git diff`. Make that review easy: small diffs, no drive-by
  reformatting or unrelated cleanups, and a short summary of what changed and why.
- **Do not add AI co-authorship trailers to commits.** No `Co-Authored-By:` line for Claude /
  Copilot / Cursor / any other tool, and no bot as the commit author. The human opening the PR is
  the sole author, takes full responsibility for the contents, and must be able to explain and
  defend every line. Disclose the tools used and the scope of their assistance in the PR
  description instead.
- **Report what was and was not verified.** CI cannot exercise real sound cards, monitors, or
  trigger hardware. State plainly which behavior was checked by tests, which by running locally
  (and on what OS / backend), and which is untested reasoning.

## Keep changes small when possible

A big diff is a cost, not an accomplishment. Aim for the smallest change that delivers the needed
behavior; anything else can be added later, when something actually needs it.

- **One concern per PR.** Bug fixes, refactors, and performance work go in separate PRs, even when
  they touch the same file.
- **Sketch before you generate.** For anything beyond a local fix, summarize the design in a few
  sentences for the human and confirm it is the smallest one that works before writing code.
- **Stop and re-plan at roughly 150 new lines of implementation.** Growth past that, or reaching
  for a new module, test file, or dependency, means the design is too elaborate. Stop, tell the
  human, and offer the smaller version.
- **YAGNI: do not build what the change does not yet need.** No speculative options, config keys,
  fallbacks, retry logic, or extension points unless a test in the same PR fails without them.
  New public API (parameters, `SOUND_CARD_*` / config keys, methods) needs a demonstrated user
  need, because it is documented and supported forever.
- **DRY: reuse or refactor before writing something new.** Grep `expyfun/_utils.py` and the
  surrounding module for existing helpers first. When code is almost reusable, factor the shared
  part into a private helper and call it from both places instead of copying and editing.
  For example, the sound backends share `_find_device` from `_sounddevice.py` rather than each
  having their own.
- **Implement at the highest layer that already exists.** Behavior shared by all sound backends
  belongs in `SoundCardController`; behavior shared by all shapes belongs in `_Triangular`.
- **Never add a dependency or promote an optional one to required** without asking first.
- **Add to an existing test before writing a new one.** Extend a test that already builds the
  objects you need (parametrized ones especially, e.g. over `_SOUND_CARD_ACS` in `conftest.py`)
  before adding a new test function, and a new test file is a signal to stop and look.

## Common commands

Use the existing environment rather than creating a new one; install in editable mode:
```bash
pip install -e . --group dev
pre-commit install --install-hooks
```

Lint / format (ruff, ruff-format, codespell, yamllint, rstcheck via pre-commit):
```bash
pre-commit run -a
```

Run tests (pytest settings live in `pyproject.toml`; warnings are errors and each test has a 15 s
timeout, so keep new tests fast):
```bash
pytest expyfun
pytest expyfun/tests/test_experiment_controller.py -k test_play --verbose
```

Useful environment variables for tests (config values are read from the environment before
`~/.expyfun/expyfun.json`):

- `_EXPYFUN_SILENT=true`: use pyglet's silent audio driver (headless Linux CI).
- `SOUND_CARD_BACKEND`, `SOUND_CARD_API`, `SOUND_CARD_NAME`, `SOUND_CARD_FS`: pick the audio
  device / backend (see `.github/workflows/tests.yml` for the per-OS CI values).
- `_EXPYFUN_WIN_INVISIBLE=true`: set by `conftest.py` so test windows are not shown.

Docs (Sphinx + sphinx-gallery, in `doc/`): `make -C doc html`.

## Architecture

- `expyfun/_experiment_controller.py`: `ExperimentController`. Window setup and `flip()`
  (pyglet event loop dispatched manually), audio/trigger/input controller wiring, data/log files.
- `expyfun/_sound_controllers/`: `SoundCardController` (`_sound_controller.py`) handles trigger
  channels, background noise, and backend selection. Each backend module (`_rtmixer.py`,
  `_sounddevice.py`, `_pyglet.py`) defines a `SoundPlayer` class with the same interface, a
  `_PRIORITY` used by `"auto"` selection, and optionally `_abort_all_queues`. Mixers / streams
  are cached per `(fs, n_channels, api, name)` in module-level registries.
- `expyfun/_tdt_controller.py`, `_trigger_controllers.py`, `_input_controllers.py`,
  `_eyelink_controller.py`: other hardware.
- `expyfun/visual/_visual.py`: shapes, text, images, and video. Shapes use OpenGL 3.2 core
  (`#version 150 core` shaders) via `pyglet.gl` directly; text and images use pyglet objects.
- `expyfun/stimuli/`, `expyfun/analyze/`, `expyfun/io/`: offline stimulus generation, analysis,
  and file I/O (no hardware).

## Code conventions (beyond what ruff enforces)

- Match the surrounding code: comment density, naming, idioms, and docstring style (numpydoc,
  `str | None` rather than "str or None").
- Nothing in the per-trial / per-frame path (`flip`, `draw`, `play`, `stop`, trigger stamping)
  may print, log at info level, allocate GPU resources, compile shaders, or query GL state that
  could stall the pipeline. Do expensive setup once at construction time.
- OpenGL objects (programs, buffers, VAOs, textures) that are created must have a clear owner and
  lifetime; do not create them inside methods that users call repeatedly.
- Audio buffers handed to a backend must not be mutated afterward, and callers' arrays must not be
  modified in place.
- Optional dependencies (rtmixer, sounddevice, mne, pandas, tdtpy, ...) are imported lazily or in
  backend modules that `_import_backend` can skip, never at the top level of `expyfun/__init__.py`.
- Workarounds for old versions of dependencies are marked with a `# TODO VERSION` comment naming
  the version at which they can be removed.
- Code adapted from an outside source must be BSD-compatible (BSD, MIT, ISC, Apache-2.0, public
  domain) and attributed in a comment directly above it, naming the source and license.
