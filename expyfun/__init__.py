"""Experimental functions
"""

__version__ = '0.1.git'

# have to import verbose first since it's needed by many things
from .utils import set_log_level, verbose_dec, set_config, \
                   get_config, get_config_path

from .experiment_controller import ExperimentController
from .eyelink_controller import EyelinkController
from .create_system_config import create_system_config

# initialize logging
set_log_level(None, False)
