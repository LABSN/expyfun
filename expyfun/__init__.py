"""
Experiment control
==================

Experiment control functions.
"""

from ._version import __version__

# have to import verbose first since it's needed by many things
from ._utils import (set_log_level, set_log_file, set_config, check_units,
                     get_config, get_config_path, fetch_data_file,
                     run_subprocess, verbose_dec as verbose, building_doc,
                     known_config_types)
from ._git import assert_version, download_version
from ._experiment_controller import ExperimentController, get_keyboard_input
from ._eyelink_controller import EyelinkController
from ._sound_controllers import SoundCardController
from ._trigger_controllers import (decimals_to_binary, binary_to_decimals,
                                   ParallelTrigger)
from ._tdt_controller import TDTController
from . import analyze
from . import codeblocks
from . import io
from . import stimuli
from . import _fixes

# INIT LOGGING
set_log_level(None, False)
set_log_file()
