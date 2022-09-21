"""Data structures and functions to evaluate the output of a coreference system.
"""

from mica_text_coref.coref.seq_coref import acceleration
from mica_text_coref.coref.seq_coref import data
from mica_text_coref.coref.seq_coref import utils

from scorch import scores
import subprocess
import re
import tempfile
import torch

class Metric:
    """Data structure of a metric for precision and recall"""

    def __init__(self, recall=None, precision=None,
                 recall_numer=None, recall_denom=None,
                 precision_numer=None, precision_denom=None) -> None:
        """Initialize using numerator and denominator, or directly from their
        ratio. If you provide recall (precision), then recall_numer
        (precision_numer) and recall_denom (precision_denom) are ignored.
        """
        # Recall
        if recall is not None:
            self.recall = recall
        elif recall_numer is not None and recall_denom is not None and (
             recall_denom != 0):
            self.recall = recall_numer/recall_denom
        else:
            self.recall = 0
        
        # Precision
        if precision is not None:
            self.precision = precision
        elif precision_numer is not None and precision_denom is not None and (
             precision_denom != 0):
            self.precision = precision_numer/precision_denom
        else:
            self.precision = 0

        # F1
        self.f1 = 0 if (self.precision == 0 and self.recall == 0) else (
            2 * self.precision * self.recall / (self.precision + self.recall))

    @property
    def score(self) -> "Metric":
        return self
    
    def __repr__(self) -> str:
        return (f"P = {100*self.precision:.1f}, R = {100*self.recall:.1f}, "
                f"F1 = {100*self.f1:.1f}")

class CoreferenceMetric:
    """Data structure for coreference metrics MUC, Bcub, CEAF, and mention."""

    def __init__(self, muc: Metric, b3: Metric, ceafe: Metric,
     ceafm: Metric, mention: Metric) -> None:
        self.muc = muc
        self.b3 = b3
        self.ceafe = ceafe
        self.ceafm = ceafm
        self.mention = mention
    
    @property
    def score(self) -> Metric:
        average_precision = (self.muc.precision + self.b3.precision +
                             self.ceafe.precision)/3
        average_recall = (self.muc.recall + self.b3.recall +
                          self.ceafe.recall)/3
        average_metric = Metric(precision=average_precision,
                                recall=average_recall)
        return average_metric
    
    def __repr__(self) -> str:
        average_f1 = (self.muc.f1 + self.b3.f1 + self.ceafe.f1)/3
        desc = (f"MUC: {self.muc}\nB3: {self.b3}\nCEAFe: {self.ceafe}\n"
                f"Average F1: {100*average_f1:.1f}\nMention: {self.mention}")
        return desc

def convert_tensor_to_cluster(tensor: torch.LongTensor) -> set[data.Mention]:
    """Find the set of mentions from the annotated tensor"""
    cluster: set[data.Mention] = set()
    n = len(tensor)
    for i in torch.where(tensor == 1)[0]:
        i = i.item()
        j = i + 1
        while j < n and tensor[j] == 2:
            j += 1
        mention = data.Mention(i, j - 1)
        cluster.add(mention)
    return cluster

def convert_to_conll(doc_id: int,
                     clusters: list[set[data.Mention]]) -> list[str]:
    """Create conll lines from clusters.

    Args:
        doc_id: Document identifier, an integer.
        clusters: List of cluster objects. 
            Each cluster is a set of data.Mention objects.
    
    Returns:
        List of lines in conll-format. Each line contains the word and
        coreference tag.
    """
    total_n_tokens = 1 + max(mention.end for cluster in clusters
                                         for mention in cluster)
    coref_column = ["-" for _ in range(total_n_tokens)]
    mentions = [(mention.begin, mention.end, i + 1)
                for i, cluster in enumerate(clusters) for mention in cluster]
    non_unigram_mentions_sorted_by_begin = sorted(
        filter(lambda mention: mention[1] - mention[0] > 0, mentions),
        key=lambda mention: (mention[0], -mention[1]))
    non_unigram_mentions_sorted_by_end = sorted(
        filter(lambda mention: mention[1] - mention[0] > 0, mentions),
        key=lambda mention: (mention[1], -mention[0]))
    unigram_mentions = filter(lambda mention: mention[1] == mention[0],
                            mentions)
    
    for begin, _, cluster_index in non_unigram_mentions_sorted_by_begin:
        if coref_column[begin] == "-":
            coref_column[begin] = "(" + str(cluster_index)
        else:
            coref_column[begin] += "|(" + str(cluster_index)
    
    for begin, _, cluster_index in unigram_mentions:
        if coref_column[begin] == "-":
            coref_column[begin] = "(" + str(cluster_index) + ")"
        else:
            coref_column[begin] += "|(" + str(cluster_index) + ")"
    
    for _, end, cluster_index in non_unigram_mentions_sorted_by_end:
        if coref_column[end] == "-":
            coref_column[end] = str(cluster_index) + ")"
        else:
            coref_column[end] += "|" + str(cluster_index) + ")"
    
    conll_lines = [f"#begin document doc_{doc_id}; part 000\n"]
    filler = "  -" * 7
    for j in range(total_n_tokens):
        line = (f"doc_{doc_id} 0 {j:>6} --{filler} {coref_column[j]}\n")
        conll_lines.append(line)
    conll_lines.append("\n")
    conll_lines = conll_lines[:-1] + ["#end document\n"]

    return conll_lines

def evaluate_coreference_perl(
    perl_scorer: str,
    groundtruth: dict[int, list[set[data.Mention]]],
    predictions: dict[int, list[set[data.Mention]]],
    verbose=False
    ) -> CoreferenceMetric:
    """Evaluates the predictions against the groundtruth annotations using
    the official conll-2012 perl scorer.

    Args:
        perl_scorer: Path to the official perl script scorer.
        groundtruth: A dictionary of list of groundtruth coreference clusters 
            (set of data.Mention objects) keyed by the doc id.
        predictions: A dictionary of list of predicted coreference clusters
            (set of data.Mention objects) keyed by the doc id.
        verbose: set to true for verbose output
    
    Return:
        CoreferenceMetric. This contains scores for MUC, B3, CEAFe, CEAFm, and
        mention.
    """
    logger = acceleration.logger
    gold_conll_lines, pred_conll_lines = [], []
    
    for doc_id, gold_clusters in groundtruth.items():
        pred_clusters = predictions[doc_id] if doc_id in predictions else []
        gold_document_conll_lines = convert_to_conll(doc_id, gold_clusters)
        pred_document_conll_lines = convert_to_conll(doc_id, pred_clusters)
        gold_conll_lines.extend(gold_document_conll_lines)
        pred_conll_lines.extend(pred_document_conll_lines)
    
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as gold_file, \
        tempfile.NamedTemporaryFile(mode="w", delete=False) as pred_file:
        gold_file.writelines(gold_conll_lines)
        pred_file.writelines(pred_conll_lines)

        if verbose:
            logger.info(f"Gold file = {gold_file.name}")
            logger.info(f"Pred file = {pred_file.name}")

        logger.info("Running the official perl conll-2012 evaluation script")
        cmd = [perl_scorer, "all", gold_file.name, pred_file.name,
                "none"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        stdout, stderr = process.communicate()
        process.wait()
        stdout = stdout.decode("utf-8")

        if verbose:
            if stderr is not None:
                logger.info(stderr)
            if stdout:
                logger.info("Official result")
                logger.info(stdout)

        matched_tuples = re.findall(
            r"Coreference: Recall: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\s+"
            r"Precision: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\s+F1:"
            r" ([0-9.]+)%", stdout, flags=re.DOTALL)
        
        muc = Metric(recall=float(matched_tuples[0][0])/100,
                     precision=float(matched_tuples[0][1])/100)
        b3 = Metric(recall=float(matched_tuples[1][0])/100,
                    precision=float(matched_tuples[1][1])/100)
        ceafm = Metric(recall=float(matched_tuples[2][0])/100,
                       precision=float(matched_tuples[2][1])/100)
        ceafe = Metric(recall=float(matched_tuples[3][0])/100,
                       precision=float(matched_tuples[3][1])/100)
        
        mention_match = re.search(
            r"Mentions: Recall: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\s+Precision:"
            r" \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\s+F1: ([0-9.]+)%", stdout,
            flags=re.DOTALL)
        mention_metric = Metric(recall=float(mention_match.group(1))/100,
                                precision=float(mention_match.group(2))/100)
        official_metric = CoreferenceMetric(muc, b3, ceafe, ceafm,
                                            mention_metric)
        return official_metric

def evaluate_coreference_scorch(
    groundtruth: dict[int, list[set[data.Mention]]],
    predictions: dict[int, list[set[data.Mention]]]
    ) -> CoreferenceMetric:
    """Evaluates two dictionaries of coreference clusters python scorch package.

    Args:
        groundtruth: A dictionary of list of groundtruth coreference clusters
                     (set of data.Mention objects) keyed by the doc id.
        predictions: A dictionary of list of predicted coreference clusters
                     (set of data.Mention objects) keyed by the doc id.
    
    Return:
        CoreferenceMetric. 
        This contains scores for MUC, B3, CEAFe, CEAFm, and mention.
    """
    logger = acceleration.logger
    gold_clusters: list[set[tuple[int, data.Mention]]] = []
    pred_clusters: list[set[tuple[int, data.Mention]]] = []
    gold_mentions: set[tuple[int, data.Mention]] = set()
    pred_mentions: set[tuple[int, data.Mention]] = set()
    doc_keys: set[int] = set()

    for doc_key, clusters in groundtruth.items():
        doc_keys.add(doc_key)
        for cluster in clusters:
            gold_cluster: set[tuple[int, data.Mention]] = set()
            for mention in cluster:
                gold_cluster.add((doc_key, mention))
                gold_mentions.add((doc_key, mention))
            gold_clusters.append(gold_cluster)

    for doc_key, clusters in predictions.items():
        if doc_key in doc_keys:
            for cluster in clusters:
                pred_cluster: set[tuple[int, data.Mention]] = set()
                for mention in cluster:
                    pred_cluster.add((doc_key, mention))
                    pred_mentions.add((doc_key, mention))
                pred_clusters.append(pred_cluster)

    logger.info("Calculating scorch:MUC")
    muc_recall, muc_precision, _ = scores.muc(gold_clusters, pred_clusters)
    logger.info("Calculating scorch:B3")
    b3_recall, b3_precision, _ = scores.b_cubed(gold_clusters, pred_clusters)
    logger.info("Calculating scorch:CEAFe")
    ceafe_recall, ceafe_precision, _ = scores.ceaf_e(gold_clusters,
                                                     pred_clusters)
    logger.info("Calculating scorch:CEAFm")
    ceafm_recall, ceafm_precision, _ = scores.ceaf_m(gold_clusters,
                                                     pred_clusters)
    n_common_mentions = len(gold_mentions.intersection(pred_mentions))
    mention_recall = 0 if len(gold_mentions) == 0 else n_common_mentions/(
        len(gold_mentions))
    mention_precision = 0 if len(pred_mentions) == 0 else n_common_mentions/(
        len(pred_mentions))

    muc = Metric(recall=muc_recall, precision=muc_precision)
    b3 = Metric(recall=b3_recall, precision=b3_precision)
    ceafe = Metric(recall=ceafe_recall, precision=ceafe_precision)
    ceafm = Metric(recall=ceafm_recall, precision=ceafm_precision)
    mention_metric = Metric(recall=mention_recall, precision=mention_precision)
    scorch_metric = CoreferenceMetric(muc, b3, ceafe, ceafm, mention_metric)
    return scorch_metric

def evaluate_coreference(
    groundtruth: torch.LongTensor, 
    predictions: torch.LongTensor,
    attn_mask: torch.FloatTensor,
    doc_ids: torch.IntTensor,
    perl_scorer: str,
    backend="perl"
    ) -> CoreferenceMetric:
    """Evaluate coreference.

    Args:
        groundtruth: LongTensor of label ids.
        predictions: LongTensor of model predictions.
        attn_mask: FloatTensor of attention mask.
        doc_ids: IntTensor of document ids.
        perl_scorer: Filepath to the perl scorer script.
        evaluation_strategy: A string indicating which backend to use for
            evaluation. It can be "perl" or "scorch".
    
    Return:
        CoreferenceMetric.
    """
    groundtruth_doc_id_to_clusters: dict[int, list[set[data.Mention]]] = {}
    predictions_doc_id_to_clusters: dict[int, list[set[data.Mention]]] = {}

    with utils.timer("tensor to cluster conversion"):
        for doc_id, gt_tensor, pred_tensor, attn in zip(
            doc_ids, groundtruth, predictions, attn_mask):
            gt_cluster = convert_tensor_to_cluster(gt_tensor[attn == 1.])
            pred_cluster = convert_tensor_to_cluster(pred_tensor[attn == 1.])
            doc_id = doc_id.item()
            if len(gt_cluster):
                if doc_id not in groundtruth_doc_id_to_clusters:
                    groundtruth_doc_id_to_clusters[doc_id] = []
                groundtruth_doc_id_to_clusters[doc_id].append(gt_cluster)
            if len(pred_cluster):
                if doc_id not in predictions_doc_id_to_clusters:
                    predictions_doc_id_to_clusters[doc_id] = []
                predictions_doc_id_to_clusters[doc_id].append(pred_cluster)
    
    if backend == "scorch":
        coref_metric = evaluate_coreference_scorch(
            groundtruth_doc_id_to_clusters, predictions_doc_id_to_clusters)
    else:
        coref_metric = evaluate_coreference_perl(
            perl_scorer, groundtruth_doc_id_to_clusters,
            predictions_doc_id_to_clusters, verbose=True)
    return coref_metric

def evaluate_sequence(
    groundtruth: torch.LongTensor,
    predictions: torch.LongTensor,
    attn: torch.FloatTensor
    ) -> Metric:
    """Evaluate two sequences token-wise.

    Args:
        groundtruth: LongTensor of label ids.
        predictions: LongTensor of prediction ids.
        attn: FloatTensor of attention mask.
    
    Returns:
        Metric.
    """
    true = groundtruth[attn == 1]
    pred = predictions[attn == 1]
    x = ((true != 0) & (true == pred)).sum().item()
    y = (true != 0).sum().item()
    z = (pred != 0).sum().item()
    metric = Metric(recall_numer=x, recall_denom=y, precision_numer=x,
                    precision_denom=z)
    return metric

def evaluate(
    groundtruth: torch.LongTensor, 
    predictions: torch.LongTensor,
    attn_mask: torch.FloatTensor,
    doc_ids: torch.IntTensor,
    perl_scorer: str,
    evaluation_strategy: str,
    ) -> CoreferenceMetric | Metric:
    """Evaluation sequence or coreference.

    Args:
        groundtruth: LongTensor of label ids.
        predictions: LongTensor of prediction ids.
        attn_mask: FloatTensor of attention mask.
        doc_ids: IntTensor of document ids.
        perl_scorer: Filepath to the perl scorer script.
        evaluation_strategy: A string indicating what kind of evaluation to run.
            It can be "perl", "scorch", or "seq".
    
    Returns:
        A coreference metric (evaluation_strategy is "perl" or "scorch") or
        metric (evaluation_strategy is "seq").
    """
    if evaluation_strategy == "seq":
        return evaluate_sequence(groundtruth, predictions, attn_mask)
    else:
        return evaluate_coreference(groundtruth, predictions, attn_mask,
                                    doc_ids, perl_scorer, evaluation_strategy)