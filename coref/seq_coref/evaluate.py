"""Functions to evaluate the output of a coreference system using the official
conll-2012 perl scorer.
"""

from mica_text_coref.coref.seq_coref import data

from scorch import scores
import subprocess
import re
import tempfile


class Metric:

    def __init__(self, recall, precision) -> None:
        self.recall = recall
        self.precision = precision
        self.f1 = 0 if (self.precision == 0 and self.recall == 0) else (
            2 * self.precision * self.recall / (self.precision + self.recall))

class CoreferenceMetric:

    def __init__(self, muc: Metric, b3: Metric, ceafe: Metric,
     ceafm: Metric, mention: Metric) -> None:
        self.muc = muc
        self.b3 = b3
        self.ceafe = ceafe
        self.ceafm = ceafm
        self.mention = mention

def convert_to_conll(doc_key_with_part_id: str, sentences: list[list[str]],
                    clusters: list[set[data.Mention]]) -> list[str]:
    """Create conll lines from clusters.

    Args:
        doc_key_with_part_id: The doc key in the jsonlines file.
        sentences: List of sentence. Each sentence is a list of tokens (string).
        clusters: List of cluster. Each cluster is a set of data.Mention
        objects.
    
    Returns:
        List of lines in conll-format. Each line contains the word and
        coreference tag.
    """
    match = re.match(r"(.+)_([^_]+)$", doc_key_with_part_id)
    doc_key, part_id = match.group(1), match.group(2)
    total_n_tokens = sum(len(sentence) for sentence in sentences)
    max_token_length = max(len(token) for sentence in sentences
                            for token in sentence)
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
    
    conll_lines = [f"#begin document {doc_key}; part {part_id.zfill(3)}\n"]
    filler = "  -" * 7
    i = 0
    for sentence in sentences:
        for j, token in enumerate(sentence):
            line = (f"{doc_key} {part_id} {j:>2} "
                   f"{token:>{max_token_length}}{filler} {coref_column[i]}\n")
            conll_lines.append(line)
            i += 1
        conll_lines.append("\n")
    conll_lines = conll_lines[:-1] + ["#end document\n"]

    return conll_lines

def evaluate_coreference(official_scorer: str, corpus: data.CorefCorpus,
    predictions: dict[str, list[set[data.Mention]]], verbose=False) -> (
        CoreferenceMetric):
    """Evaluates the predictions against the groundtruth annotations using
    the official conll-2012 perl scorer.

    Args:
        official_scorer: Path to the official perl script scorer.
        corpus: Coreference corpus annotated with groundtruth annotations.
        predictions: A dictionary of list of coreference clusters (set of
            data.Mention objects) keyed by the doc id.
        verbose: set to true for verbose output
    
    Return:
        CoreferenceMetric. This contains scores for MUC, B3, CEAFe, CEAFm, and
        mention.
    """
    gold_conll_lines, pred_conll_lines = [], []
    
    for document in corpus.documents:
        doc_key = document.doc_key
        gold_clusters = document.clusters
        pred_clusters = predictions[doc_key] if doc_key in predictions else []
        gold_document_conll_lines = convert_to_conll(
            doc_key, document.sentences, gold_clusters)
        pred_document_conll_lines = convert_to_conll(
            doc_key, document.sentences, pred_clusters)
        gold_conll_lines.extend(gold_document_conll_lines)
        pred_conll_lines.extend(pred_document_conll_lines)
    
    with tempfile.NamedTemporaryFile(mode="w", delete=True) as gold_file, \
        tempfile.NamedTemporaryFile(mode="w", delete=True) as pred_file:
        gold_file.writelines(gold_conll_lines)
        pred_file.writelines(pred_conll_lines)

        if verbose:
            print(f"Gold file = {gold_file.name}")
            print(f"Pred file = {pred_file.name}")

        cmd = [official_scorer, "all", gold_file.name, pred_file.name,
                "none"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        stdout, stderr = process.communicate()
        process.wait()
        stdout = stdout.decode("utf-8")

        if verbose:
            if stderr is not None:
                print(stderr)
            if stdout:
                print("Official result")
                print(stdout)

        matched_tuples = re.findall(
            r"Coreference: Recall: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\s+"
            r"Precision: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\s+F1:"
            r" ([0-9.]+)%", stdout, flags=re.DOTALL)
        
        muc = Metric(float(matched_tuples[0][0]), float(matched_tuples[0][1]))
        b3 = Metric(float(matched_tuples[1][0]), float(matched_tuples[1][1]))
        ceafm = Metric(float(matched_tuples[2][0]), float(matched_tuples[2][1]))
        ceafe = Metric(float(matched_tuples[3][0]), float(matched_tuples[3][1]))
        
        mention_match = re.search(
            r"Mentions: Recall: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\s+Precision:"
            r" \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\s+F1: ([0-9.]+)%", stdout,
            flags=re.DOTALL)
        mention_metric = Metric(float(mention_match.group(1)),
                                float(mention_match.group(2)))
        official_metric = CoreferenceMetric(muc, b3, ceafe, ceafm,
                                            mention_metric)
        return official_metric

def evaluate_coreference_scorch(corpus: data.CorefCorpus,
    predictions: dict[str, list[set[data.Mention]]]) -> CoreferenceMetric:
    """Evaluates the predictions against the groundtruth annotations using the
    unofficial python scorch package.

    Args:
        corpus: Coreference corpus annotated with groundtruth annotations.
        predictions: A dictionary of list of coreference clusters (set of
            data.Mention objects) keyed by the doc id.
    
    Return:
        CoreferenceMetric. This contains scores for MUC, B3, CEAFe, CEAFm, and
        mention.
    """
    gold_clusters: list[set[tuple[str, data.Mention]]] = []
    pred_clusters: list[set[tuple[str, data.Mention]]] = []
    gold_mentions: set[tuple[str, data.Mention]] = set()
    pred_mentions: set[tuple[str, data.Mention]] = set()
    doc_keys: set[str] = set()

    for document in corpus.documents:
        doc_keys.add(document.doc_key)
        for cluster in document.clusters:
            gold_cluster: set[tuple[str, data.Mention]] = set()
            for mention in cluster:
                gold_cluster.add((document.doc_key, mention))
                gold_mentions.add((document.doc_key, mention))
            gold_clusters.append(gold_cluster)
    
    for doc_key, clusters in predictions.items():
        if doc_key in doc_keys:
            for cluster in clusters:
                pred_cluster: set[tuple[str, data.Mention]] = set()
                for mention in cluster:
                    pred_cluster.add((doc_key, mention))
                    pred_mentions.add((doc_key, mention))
                pred_clusters.append(pred_cluster)
    
    muc_recall, muc_precision, _ = scores.muc(gold_clusters, pred_clusters)
    b3_recall, b3_precision, _ = scores.b_cubed(gold_clusters, pred_clusters)
    ceafe_recall, ceafe_precision, _ = scores.ceaf_e(gold_clusters,
                                                    pred_clusters)
    ceafm_recall, ceafm_precision, _ = scores.ceaf_m(gold_clusters,
                                                    pred_clusters)
    n_common_mentions = len(gold_mentions.intersection(pred_mentions))
    mention_recall = 0 if len(gold_mentions) == 0 else n_common_mentions/(
        len(gold_mentions))
    mention_precision = 0 if len(pred_mentions) == 0 else n_common_mentions/(
        len(pred_mentions))

    muc = Metric(muc_recall, muc_precision)
    b3 = Metric(b3_recall, b3_precision)
    ceafe = Metric(ceafe_recall, ceafe_precision)
    ceafm = Metric(ceafm_recall, ceafm_precision)
    mention_metric = Metric(mention_recall, mention_precision)
    scorch_metric = CoreferenceMetric(muc, b3, ceafe, ceafm, mention_metric)
    return scorch_metric