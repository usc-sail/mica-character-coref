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
        return (f"P={100*self.precision:3.1f} R={100*self.recall:3.1f} "
                f"F1={100*self.score:3.1f}")

class CharacterRecognitionTrainer(Trainer):

    def evaluate(self, **tensors) -> Metric:
        pos_label = 0
        logits, labels, offset = (
            tensors["logits"], tensors["labels"], tensors["token_offset"])
        pred = logits.argmax(dim=2)
        mask = ~((offset[:,:,0] == 0) & (offset[:,:,1] == 0))
        pred = pred[mask]
        labels = labels[mask]
        tp = ((labels == pred) & (pred == pos_label)).sum().item()
        fp = ((labels != pred) & (pred == pos_label)).sum().item()
        fn = ((labels != pred) & (labels == pos_label)).sum().item()
        precision = tp/(tp + fp + 1e-23)
        recall = tp/(tp + fn + 1e-23)
        return CharacterRecognitionMetric(precision, recall)