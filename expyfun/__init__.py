"""Experiment control functions
"""

__version__ = '2.0.0.git'

# have to import verbose first since it's needed by many things
from ._utils import (set_log_level, set_log_file, set_config, check_units,
                     get_config, get_config_path, fetch_data_file)
from ._utils import verbose_dec as verbose
from ._experiment_controller import (ExperimentController, wait_secs,
                                     get_keyboard_input)
from ._eyelink_controller import EyelinkController
from . import analyze
from . import codeblocks
from . import io
from . import stimuli

# INIT LOGGING
set_log_level(None, False)
set_log_file()
