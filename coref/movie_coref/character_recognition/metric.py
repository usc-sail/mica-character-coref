"""Evaluation Metric for character recognition.
"""

from mica_text_coref.coref.movie_coref.trainer import Metric

class CharacterRecognitionMetric(Metric):

    def __init__(self, precision, recall) -> None:
        super().__init__()
        self.precision = precision
        self.recall = recall
    
    @property
    def score(self) -> float:
        return 2*self.precision*self.recall/(self.precision+self.recall+1e-23)
    
    def __repr__(self) -> str:
        return (f"P={100*self.precision:3.1f} R={100*self.recall:3.1f} "
                f"F1={100*self.score:3.1f}")