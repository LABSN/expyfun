"""Experimental functions
"""

__version__ = '0.1.git'

# have to import verbose first since it's needed by many things
from .utils import set_log_level, set_log_file, verbose, set_config, \
                   get_config, get_config_path

from .experiment_controller import ExperimentController
from .eyelink_controller import EyelinkController

# initialize logging
set_log_level(None, False)
set_log_file()

