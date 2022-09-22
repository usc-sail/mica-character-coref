"""Intialize accelerator object"""

from absl import flags
import accelerate
from accelerate import logging as alogging
import logging

FLAGS = flags.FLAGS
mixed_precision = "fp16" if FLAGS.use_mixed_precision else "no"
accelerator = accelerate.Accelerator(
    gradient_accumulation_steps=FLAGS.grad_accumulation_steps,
    mixed_precision=mixed_precision)
logger = alogging.get_logger("")