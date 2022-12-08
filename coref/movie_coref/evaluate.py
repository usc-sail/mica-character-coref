"""Evaluate output of movie coreference model"""
from mica_text_coref.coref.movie_coref import data
from mica_text_coref.coref.movie_coref import conll

import numpy as np
import os
import tempfile

def _evaluate_coreference(reference_scorer: str, gold_conll_lines: str, pred_conll_lines: str,
                          gold_file: str, pred_file: str) -> data.CorefMetric:
    """Evaluate coreference using the official perl conll-2012 scorer.

    Args:
        reference_scorer: Filepath of the scorer
        gold_conll_lines: CONLL lines of the ground truth
        pred_conll_lines: CONLL lines of the model output
        gold_file: Filepath where the gold CONLL file is saved
        pred_file: Filepath where the pred CONLL file is saved
    
    Returns:
        CorefMetric: A python object containing coreference metric scores (muc, bcub, ceafe)
    """
    result = conll.evaluate_conll(reference_scorer, gold_conll_lines, pred_conll_lines, gold_file, pred_file)
    metric = data.CorefMetric()
    metric.muc = data.Metric(*result["muc"]["all"])
    metric.bcub = data.Metric(*result["bcub"]["all"])
    metric.ceafe = data.Metric(*result["ceafe"]["all"])
    return metric

def _evaluate_character_heads(gold: list[int], pred: list[int]) -> data.Metric:
    """Evaluate character head prediction.

    Args:
        gold: List of gold character head labels. Each list value is 0 or 1.
        pred: List of predicted character head values from the model.

    Return:
        Metric: A python object containing precision, recall, and F1 scores.
    """
    assert len(gold) == len(pred)
    gold = np.array(gold)
    pred = np.array(pred)
    tp = ((gold == 1) & (pred == 1)).sum()
    fp = ((gold != 1) & (pred == 1)).sum()
    fn = ((gold == 1) & (pred != 1)).sum()
    precision = float(100*tp/(tp + fp + 1e-23))
    recall = float(100*tp/(tp + fn + 1e-23))
    return data.Metric(precision, recall)

def evaluate(documents: list[data.CorefDocument], results: list[data.CorefResult], reference_scorer: str, 
             output_dir: str = None, filename: str = "dev") -> data.MovieCorefMetric:
    """Evaluate output of movie coreference model using the official perl conll-2012 scorer.
    
    CONLL files are written to 'output_dir' with 'filename' included in the file names.
    If 'output_dir' is NONE, then temporary directories are used that are immediately deleted after.

    Args:
        documents: List of CorefDocument objects. Used to obtain gold cluster and character head values.
        results: List of CorefResult objects. Used to obtain the predicted clusters and character heads.
        reference_scorer: Filepath of the official perl conll-2012 scorer.
        output_dir: Directory where CONLL files are written. If NONE, temporary directories are used.
        filename: Text to be included in the CONLL filenames.
    
    Returns:
        MovieCorefMetric: A python object containing coreference metric (muc, bcub, and ceafe) scores
            for word-based and span-based clusters, and character head classification performance
            score (precision, recall, and f1)
    """
    # create the gold/pred word/span conll lines, and gold/pred character head ids
    assert len(documents) == len(results), ("documents (list[CorefDocument]) and results (list[CorefResult]) should be"
                                            " of same length")
    gold_word_conll_lines, gold_span_conll_lines, gold_character_ids = [], [], []
    pred_word_conll_lines, pred_span_conll_lines, pred_character_ids = [], [], []
    for document, result in zip(documents, results):
        gold_word_clusters = [set((mention.head, mention.head) for mention in cluster)
                                for cluster in document.clusters.values()]
        gold_span_clusters = [set((mention.begin, mention.end) for mention in cluster)
                                for cluster in document.clusters.values()]
        pred_word_clusters = [set((word, word) for word in cluster) for cluster in result.predicted_word_clusters]
        pred_span_clusters = result.predicted_span_clusters
        gold_character_ids += document.word_head_ids
        pred_character_ids += result.predicted_character_heads.tolist()
        gold_word_conll_lines += conll.convert_to_conll(document, gold_word_clusters)
        gold_span_conll_lines += conll.convert_to_conll(document, gold_span_clusters)
        pred_word_conll_lines += conll.convert_to_conll(document, pred_word_clusters)
        pred_span_conll_lines += conll.convert_to_conll(document, pred_span_clusters)

    # evaluate coreference and character heads
    movie_coref_metric = data.MovieCorefMetric()
    with tempfile.TemporaryDirectory() as temp_dir:
        if output_dir is None:
            output_dir = temp_dir
        gold_word_file = os.path.join(output_dir, f"gold_epoch.word.{filename}.conll")
        gold_span_file = os.path.join(output_dir, f"gold_epoch.span.{filename}.conll")
        pred_word_file = os.path.join(output_dir, f"pred_epoch.word.{filename}.conll")
        pred_span_file = os.path.join(output_dir, f"pred_epoch.span.{filename}.conll")
        movie_coref_metric.word_coref = _evaluate_coreference(
            reference_scorer, gold_word_conll_lines, pred_word_conll_lines, gold_word_file, pred_word_file)
        movie_coref_metric.span_coref = _evaluate_coreference(
            reference_scorer, gold_span_conll_lines, pred_span_conll_lines, gold_span_file, pred_span_file)
    movie_coref_metric.character = _evaluate_character_heads(gold_character_ids, pred_character_ids)

    return movie_coref_metric