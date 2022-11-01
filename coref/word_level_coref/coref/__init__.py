""" Describes a model to extract coreferential spans from a list of tokens.

  Usage example:

  model = CorefModel("config.toml", "debug")
  model.evaluate("dev")
"""

from mica_text_coref.coref.word_level_coref.coref.coref_model import CorefModel


__all__ = [
    "CorefModel"
]
