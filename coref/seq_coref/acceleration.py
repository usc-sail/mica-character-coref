"""Intialize accelerator object"""

import accelerate
from accelerate import logging as alogging
import logging

accelerator = accelerate.Accelerator()
logger = alogging.get_logger("")
logger.logger.addHandler(logging.StreamHandler())