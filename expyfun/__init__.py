"""Experiment control functions
"""

__version__ = '1.1.0.git'

# have to import verbose first since it's needed by many things
from ._utils import (set_log_level, set_log_file, set_config,
                     get_config, get_config_path)
from ._utils import verbose_dec as verbose
from ._experiment_controller import ExperimentController, wait_secs
from ._eyelink_controller import EyelinkController
from . import analyze  # fast enough, include here
from . import codeblocks
from . import stimuli

# INIT LOGGING
set_log_level(None, False)
set_log_file()
