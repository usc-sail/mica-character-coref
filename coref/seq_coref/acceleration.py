"""Intialize accelerator object"""

from absl import flags
import accelerate
from accelerate import logging as alogging
import logging

FLAGS = flags.FLAGS
accelerator = accelerate.Accelerator(
    gradient_accumulation_steps=FLAGS.grad_accumulation_steps)
logger = alogging.get_logger("")
logger.logger.addHandler(logging.StreamHandler())