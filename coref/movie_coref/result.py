"""Data class for storing gold and predicted coreference labels and score them.
"""

from mica_text_coref.coref.movie_coref.data import CorefDocument
from mica_text_coref.coref.movie_coref import conll

import numpy as np
import os

class Metric:
    """General metric class for precision, recall, and F1
    """
    def __init__(self, precision, recall, f1=None):
        self.precision = precision
        self.recall = recall
        if f1 is None:
            self.f1 = 2 * self.precision * self.recall / (1e-23 + self.precision + self.recall)
        else:
            self.f1 = f1
    
    def __repr__(self) -> str:
        return (f"precision={self.precision:.4f} recall={self.recall:.4f} f1={self.f1:.4f}")

class CorefResult:
    """Data class to store gold and predicted coreference and character head
    labels, and compute performance using conll reference scorer.
    """

    def __init__(self, reference_scorer: str, results_dir: str, epoch: int) -> (
        None):
        """Initialize CorefResult object.

        Args:
            reference_scorer: Path to the conll perl scorer script.
            results_dir: Directory where the conll files will be saved.
            epoch: Epoch number, used in the file names.
        """
        self.reference_scorer = reference_scorer
        self.results_dir = results_dir
        self.epoch = epoch
        self._movie_to_data: dict[str, any] = {}
        self._word_metric: tuple[Metric, Metric, Metric] = None
        self._span_metric: tuple[Metric, Metric, Metric] = None
        self._character_metric: Metric = None

    def _init_result(self, document: CorefDocument):
        self._movie_to_data[document.movie] = dict(
            document=document, gold_word=[], pred_word=[], gold_character=[],
            pred_character=[], gold_span=[], pred_span=[])
    
    def _clear_scores(self):
        self._word_metric = self._span_metric = self._character_metric = None

    def add_word_clusters(
        self, document: CorefDocument, gold_word_clusters: list[set[int]], 
        pred_word_clusters: list[set[int]]):
        """Add gold and predicted word clusters of the document.
        """
        self._clear_scores()
        if document.movie not in self._movie_to_data:
            self._init_result(document)
        self._movie_to_data[document.movie]["gold_word"] = [
            set([(i, i) for i in cluster]) for cluster in gold_word_clusters]
        self._movie_to_data[document.movie]["pred_word"] = [
            set([(i, i) for i in cluster]) for cluster in pred_word_clusters]
    
    def add_span_clusters(self, document: CorefDocument,
        gold_span_clusters: list[set[tuple[int, int]]], 
        pred_span_clusters: list[set[tuple[int, int]]]):
        """Add gold and predicted span clusters of the document.
        """
        self._clear_scores()
        if document.movie not in self._movie_to_data:
            self._init_result(document)
        self._movie_to_data[document.movie]["gold_span"] = gold_span_clusters
        self._movie_to_data[document.movie]["pred_span"] = pred_span_clusters

    def add_characters(self, document: CorefDocument, 
        gold_characters: list[int], pred_characters: list[int]):
        """Add gold and predicted character heads of the document.
        """
        self._clear_scores()
        if document.movie not in self._movie_to_data:
            self._init_result(document)
        assert len(gold_characters) == len(pred_characters)
        self._movie_to_data[document.movie]["gold_character"] = gold_characters
        self._movie_to_data[document.movie]["pred_character"] = pred_characters

    def add(self, other: "CorefResult"):
        """Join another CorefResult.
        """
        self._clear_scores()
        assert set(self._movie_to_data.keys()).isdisjoint(
            set(other._movie_to_data.keys())), (
                "You are trying to join two CorefResults that already share"
                " some common document.")
        self._movie_to_data.update(other._movie_to_data)

    def _evaluate_sequence(self, gold: list[int], pred: list[int], 
        pos_label: int) -> Metric:
        assert len(gold) == len(pred)
        gold = np.array(gold)
        pred = np.array(pred)
        tp = ((gold == pos_label) & (pred == pos_label)).sum()
        fp = ((gold != pos_label) & (pred == pos_label)).sum()
        fn = ((gold == pos_label) & (pred != pos_label)).sum()
        precision = tp/(tp + fp + 1e-23)
        recall = tp/(tp + fn + 1e-23)
        return Metric(precision, recall)
    
    def _average_conll_f1(self) -> tuple[float, float]:
        word_f1 = np.mean([m.f1 for m in self._word_metric])
        span_f1 = np.mean([m.f1 for m in self._span_metric])
        return word_f1, span_f1

    def _score(self):
        if self._word_metric is None or self._span_metric is None or self._character_metric is None:
            gold_word_conll_lines, pred_word_conll_lines = [], []
            gold_span_conll_lines, pred_span_conll_lines = [], []
            gold_character, pred_character = [], []
            movies = []
            for movie, data in self._movie_to_data.items():
                movies.append(movie)
                gold_word_conll_lines.extend(conll.convert_to_conll(data["document"], 
                    data["gold_word"]))
                pred_word_conll_lines.extend(conll.convert_to_conll(data["document"], 
                    data["pred_word"]))
                gold_span_conll_lines.extend(conll.convert_to_conll(data["document"], 
                    data["gold_span"]))
                pred_span_conll_lines.extend(conll.convert_to_conll(data["document"], 
                    data["pred_span"]))
                if len(data["document"].token) == len(data["gold_character"]) == (
                        len(data["pred_character"])):
                    gold_character.extend(data["gold_character"])
                    pred_character.extend(data["pred_character"])
            name = "__".join(sorted(movies))
            epoch_dir = os.path.join(self.results_dir, f"epoch_{self.epoch}")
            os.makedirs(epoch_dir, exist_ok=True)
            gold_word_file = os.path.join(epoch_dir, f"gold.word.{name}.conll")
            pred_word_file = os.path.join(epoch_dir, f"pred.word.{name}.conll")
            gold_span_file = os.path.join(epoch_dir, f"gold.span.{name}.conll")
            pred_span_file = os.path.join(epoch_dir, f"pred.span.{name}.conll")
            (muc_precision, muc_recall, b_cubed_precision, b_cubed_recall,
            ceaf_e_precision, ceaf_e_recall) = (conll.evaluate_conll(self.reference_scorer, 
                gold_word_conll_lines, pred_word_conll_lines, gold_word_file, pred_word_file))
            self._word_metric = (Metric(muc_precision, muc_recall), Metric(b_cubed_precision, 
                b_cubed_recall), Metric(ceaf_e_precision, ceaf_e_recall))
            (muc_precision, muc_recall, b_cubed_precision, b_cubed_recall, ceaf_e_precision, 
            ceaf_e_recall) = (conll.evaluate_conll(self.reference_scorer, gold_span_conll_lines, 
                pred_span_conll_lines, gold_span_file, pred_span_file))
            self._span_metric = (Metric(muc_precision, muc_recall), Metric(b_cubed_precision, 
                b_cubed_recall), Metric(ceaf_e_precision, ceaf_e_recall))
            self._character_metric = self._evaluate_sequence(gold_character, pred_character, 1)

    def __repr__(self) -> str:
        self._score()
        word_score, span_score = self._average_conll_f1()
        d = f"Word: {word_score:.4f}, Span: {span_score:.4f}, Character: {self._character_metric}"
        return d