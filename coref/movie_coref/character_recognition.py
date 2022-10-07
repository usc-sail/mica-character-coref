"""Custom trainer and metric for character recognition
"""

from mica_text_coref.coref.movie_coref.trainer import Trainer, Metric

class CharacterRecognitionMetric(Metric):

    def __init__(self, precision, recall) -> None:
        super().__init__()
        self.precision = precision
        self.recall = recall
    
    @property
    def score(self) -> float:
        return 2*self.precision*self.recall/(self.precision+self.recall+1e-23)
    
    def __repr__(self) -> str:
        return f"P={self.precision:.2f} R={self.recall:.2f} F1={self.score:.2f}"

class CharacterRecognitionTrainer(Trainer):

    def evaluate(self, **tensors) -> Metric:
        logits, labels, offset = (
            tensors["logits"], tensors["labels"], tensors["token_offset"])
        pred = logits.argmax(dim=2)
        mask = ~((offset[:,:,0] == 0) & (offset[:,:,1] == 0))
        pred = pred[mask]
        labels = labels[mask]
        tp = ((labels == pred) & (pred != 0)).sum().item()
        fp = ((labels != pred) & (pred != 0)).sum().item()
        fn = ((labels != pred) & (labels != 0)).sum().item()
        precision = tp/(tp + fp + 1e-23)
        recall = tp/(tp + fn + 1e-23)
        return CharacterRecognitionMetric(precision, recall)