"""Custom trainer for character recognition
"""
from mica_text_coref.coref.movie_coref.character_recognition.metric import CharacterRecognitionMetric
from mica_text_coref.coref.movie_coref.trainer import Trainer, Metric

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