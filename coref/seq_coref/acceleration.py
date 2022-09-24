"""Intialize accelerator object"""

from absl import flags
import accelerate
from accelerate import logging as alogging

FLAGS = flags.FLAGS
mixed_precision = "fp16" if ("mixed_precision" in FLAGS and 
                             FLAGS.use_mixed_precision) else "no"
grad_accumulation_steps = (
    FLAGS.grad_accumulation_steps if "grad_accumulation_steps" in FLAGS else 1)
accelerator = accelerate.Accelerator(
    gradient_accumulation_steps=grad_accumulation_steps,
    mixed_precision=mixed_precision)
logger = alogging.get_logger("")