"""Experiment control functions
"""

__version__ = '1.1.0.git'

# have to import verbose first since it's needed by many things
from ._utils import (set_log_level, set_config,
                     get_config, get_config_path)
from ._utils import verbose_dec as verbose
from ._experiment_controller import ExperimentController, wait_secs
from ._eyelink_controller import EyelinkController
from ._create_system_config import create_system_config
from . import analyze  # fast enough, include here
from . import stimuli

# initialize logging
set_log_level(None, False)
