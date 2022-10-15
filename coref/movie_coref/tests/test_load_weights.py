"""Test if pretrained wl-roberta weights can be loaded.
"""

from mica_text_coref.coref.movie_coref.coreference.model import MovieCoreference

import os
import unittest

class TestLoadWeights(unittest.TestCase):
    """Testing loading weights.
    """

    def test_load_weights(self):
        model = MovieCoreference("roberta-large", 5, 5, 5, 16, 2, 256, True, 0.3)
        weights_path = os.path.join(
            os.getcwd(), 
            "data/word_level_coref/data/"
            "roberta_(e20_2021.05.02_01.16)_release.pt")
        model.load_weights(weights_path)
        self.assertTrue("Weights successfully loaded!")

if __name__=="__main__":
    unittest.main()