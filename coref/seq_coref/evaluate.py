"""Functions to evaluate the output of a coreference system using the official
conll-2012 perl scorer.
"""

from mica_text_coref.coref.seq_coref import coref_longformer
from mica_text_coref.coref.seq_coref import data
from mica_text_coref.coref.seq_coref import util

from absl import logging
import numpy as np
from scorch import scores
import subprocess
import re
import tempfile
import time
import torch
from torch.utils import data as tdata

class Metric:

    def __init__(self, recall, precision) -> None:
        self.recall = recall
        self.precision = precision
        self.f1 = 0 if (self.precision == 0 and self.recall == 0) else (
            2 * self.precision * self.recall / (self.precision + self.recall))
    
    def __repr__(self) -> str:
        return (f"P = {100*self.precision:.1f}, R = {100*self.recall:.1f}, "
                f"F1 = {100*self.f1:.1f}")

class CoreferenceMetric:

    def __init__(self, muc: Metric, b3: Metric, ceafe: Metric,
     ceafm: Metric, mention: Metric) -> None:
        self.muc = muc
        self.b3 = b3
        self.ceafe = ceafe
        self.ceafm = ceafm
        self.mention = mention
    
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

def evaluate_clusters_official(official_scorer: str,
    groundtruth: dict[int, list[set[data.Mention]]],
    predictions: dict[int, list[set[data.Mention]]],
    doc_id_to_doc_key: dict[int, str],
    doc_id_to_sentences: dict[int, list[list[str]]],
    verbose=False
    ) -> CoreferenceMetric:
    """Evaluates the predictions against the groundtruth annotations using
    the official conll-2012 perl scorer. This function will throw an error if
    any key of the groundtruth dictionary is not present in both the
    doc_id_to_doc_key and doc_id_to_sentences dictionaries.

    Args:
        official_scorer: Path to the official perl script scorer.
        groundtruth: A dictionary of list of groundtruth coreference clusters 
            (set of data.Mention objects) keyed by the doc id.
        predictions: A dictionary of list of predicted coreference clusters
            (set of data.Mention objects) keyed by the doc id.
        doc_id_to_doc_key: A map (dictionary) from doc id to doc key.
        doc_id_to_sentences: A map (dictionary) from doc id to list of 
            sentences. Each sentence is a list of string words.
        verbose: set to true for verbose output
    
    Return:
        CoreferenceMetric. This contains scores for MUC, B3, CEAFe, CEAFm, and
        mention.
    """
    gold_conll_lines, pred_conll_lines = [], []
    
    for doc_id, gold_clusters in groundtruth.items():
        doc_key = doc_id_to_doc_key[doc_id]
        sentences = doc_id_to_sentences[doc_id]
        pred_clusters = predictions[doc_id] if doc_id in predictions else []
        gold_document_conll_lines = convert_to_conll(
            doc_key, sentences, gold_clusters)
        pred_document_conll_lines = convert_to_conll(
            doc_key, sentences, pred_clusters)
        gold_conll_lines.extend(gold_document_conll_lines)
        pred_conll_lines.extend(pred_document_conll_lines)
    
    with tempfile.NamedTemporaryFile(mode="w", delete=True) as gold_file, \
        tempfile.NamedTemporaryFile(mode="w", delete=True) as pred_file:
        gold_file.writelines(gold_conll_lines)
        pred_file.writelines(pred_conll_lines)

        if verbose:
            logging.info(f"Gold file = {gold_file.name}")
            logging.info(f"Pred file = {pred_file.name}")

        logging.info("Running the official perl conll-2012 evaluation script")
        cmd = [official_scorer, "all", gold_file.name, pred_file.name,
                "none"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        stdout, stderr = process.communicate()
        process.wait()
        stdout = stdout.decode("utf-8")

        if verbose:
            if stderr is not None:
                logging.info(stderr)
            if stdout:
                logging.info("Official result")
                logging.info(stdout)

        matched_tuples = re.findall(
            r"Coreference: Recall: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\s+"
            r"Precision: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\s+F1:"
            r" ([0-9.]+)%", stdout, flags=re.DOTALL)
        
        muc = Metric(float(matched_tuples[0][0])/100,
                    float(matched_tuples[0][1])/100)
        b3 = Metric(float(matched_tuples[1][0])/100,
                    float(matched_tuples[1][1])/100)
        ceafm = Metric(float(matched_tuples[2][0])/100,
                        float(matched_tuples[2][1])/100)
        ceafe = Metric(float(matched_tuples[3][0])/100,
                        float(matched_tuples[3][1])/100)
        
        mention_match = re.search(
            r"Mentions: Recall: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\s+Precision:"
            r" \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\s+F1: ([0-9.]+)%", stdout,
            flags=re.DOTALL)
        mention_metric = Metric(float(mention_match.group(1)),
                                float(mention_match.group(2)))
        official_metric = CoreferenceMetric(muc, b3, ceafe, ceafm,
                                            mention_metric)
        return official_metric

def evaluate_tensors_official(official_scorer: str,
    groundtruth: torch.LongTensor,
    predictions: torch.LongTensor,
    doc_ids: torch.IntTensor,
    attn_mask: torch.FloatTensor,
    corpus: data.CorefCorpus,
    evaluate_against_corpus = False) -> (
        CoreferenceMetric | tuple[CoreferenceMetric, CoreferenceMetric]):
    """Evaluate the predictions against the groundtruth annotations using the
    official conll-2012 perl scorer. The groundtruth and predictions are
    represented by tensors.

    Args:
        official_scorer: Path to the official perl script scorer.
        groundtruth: Integer tensor annotated with groundtruth cluster
            mentions.
        predictions: Integer tensor annotated with predicted cluster mentions.
        doc_ids: Integer tensor containing doc ids of the corresponding
            groundtruth and predictions tensors.
        corpus: Original coreference corpus from which the groundtruth tensors
            was created.
    
    Return:
        Tuple of two CoreferenceMetric objects.
    """
    corpus_doc_id_to_clusters: dict[int, list[set[data.Mention]]] = {}
    groundtruth_doc_id_to_clusters: dict[int, list[set[data.Mention]]] = {}
    predictions_doc_id_to_clusters: dict[int, list[set[data.Mention]]] = {}
    doc_id_to_doc_key: dict[int, str] = corpus.get_doc_id_to_doc_key()
    doc_id_to_sentences: dict[int, list[list[str]]] = (
        corpus.get_doc_id_to_sentences())

    logging.info("Converting tensors to clusters")
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
    
    coref_metric1 = evaluate_clusters_official(official_scorer, 
        groundtruth_doc_id_to_clusters, predictions_doc_id_to_clusters,
        doc_id_to_doc_key, doc_id_to_sentences)
    
    if evaluate_against_corpus:
        for document in corpus.documents:
            doc_id = document.doc_id
            if len(document.clusters):
                corpus_doc_id_to_clusters[doc_id] = document.clusters
        coref_metric2 = evaluate_clusters_official(official_scorer, 
            corpus_doc_id_to_clusters, predictions_doc_id_to_clusters,
            doc_id_to_doc_key, doc_id_to_sentences)
        return coref_metric1, coref_metric2
    else:
        return coref_metric1

def evaluate_clusters_scorch(
    groundtruth: dict[int, list[set[data.Mention]]],
    predictions: dict[int, list[set[data.Mention]]]
    ) -> CoreferenceMetric:
    """Evaluates the predictions against the groundtruth annotations using the
    unofficial python scorch package.

    Args:
        groundtruth: A dictionary of list of groundtruth coreference clusters
            (set of data.Mention objects) keyed by the doc id.
        predictions: A dictionary of list of predicted coreference clusters
            (set of data.Mention objects) keyed by the doc id.
    
    Return:
        CoreferenceMetric. This contains scores for MUC, B3, CEAFe, CEAFm, and
        mention.
    """
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
    
    logging.info("Calculating scorch:MUC")
    muc_recall, muc_precision, _ = scores.muc(gold_clusters, pred_clusters)
    logging.info("Calculating scorch:B3")
    b3_recall, b3_precision, _ = scores.b_cubed(gold_clusters, pred_clusters)
    logging.info("Calculating scorch:CEAFe")
    ceafe_recall, ceafe_precision, _ = scores.ceaf_e(gold_clusters,
                                                    pred_clusters)
    logging.info("Calculating scorch:CEAFm")
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

# TODO: pass mentions argument to only evaluate against the mention which are
# not representative mentions
def evaluate_tensors_scorch(groundtruth: torch.LongTensor, 
    predictions: torch.LongTensor,
    doc_ids: torch.IntTensor,
    attn_mask: torch.FloatTensor,
    corpus: data.CorefCorpus | None = None,
    evaluate_against_corpus = False
    ) -> CoreferenceMetric | tuple[CoreferenceMetric, CoreferenceMetric]:
    """Evaluate the predictions against the groundtruth annotations using the
    unofficial python scorch library. The groundtruth and predictions are
    represented by tensors. If corpus is not None, also predict against the
    clusters of the corpus.

    Args:
        groundtruth: Integer tensor annotated with groundtruth cluster
            mentions.
        predictions: Integer tensor annotated with predicted cluster mentions.
        doc_ids: Integer tensor containing doc ids of the corresponding
            groundtruth and predictions tensors.
        corpus: Original coreference corpus from which the groundtruth tensors
            was created.
    
    Return:
        CoreferenceMetric or tuple of two CoreferenceMetric.
    """
    assert not evaluate_against_corpus or corpus is not None, (
        "Provide corpus if evaluating against corpus")
    groundtruth_doc_id_to_clusters: dict[int, list[set[data.Mention]]] = {}
    predictions_doc_id_to_clusters: dict[int, list[set[data.Mention]]] = {}

    logging.info("Converting tensors to clusters")
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
    
    coref_metric1 = evaluate_clusters_scorch(
        groundtruth_doc_id_to_clusters, predictions_doc_id_to_clusters)
    
    if evaluate_against_corpus:
        corpus_doc_id_to_clusters: dict[int, list[set[data.Mention]]] = {}
        for document in corpus.documents:
            doc_id = document.doc_id
            if len(document.clusters):
                corpus_doc_id_to_clusters[doc_id] = document.clusters
        coref_metric2 = evaluate_clusters_scorch(
            corpus_doc_id_to_clusters, predictions_doc_id_to_clusters)
        return coref_metric1, coref_metric2
    else:
        return coref_metric1

def evaluate_dataloader(model: coref_longformer.CorefLongformerModel,
    dataloader: tdata.DataLoader,
    corpus: data.CorefCorpus | None = None,
    official_scorer: str | None = None,
    use_official = False,
    batch_size = 64,
    print_n_batches = 10,
    use_dynamic_loading = False,
    evaluate_against_corpus = False
    ) -> CoreferenceMetric | tuple[CoreferenceMetric, CoreferenceMetric]:
    """
    Evaluate the trained coreference longformer model on the given dataloader,
    and return Coreference Metric.
    """
    if use_official:
        assert official_scorer is not None, "Official scorer missing!"
        assert corpus is not None, "Corpus missing!"

    if evaluate_against_corpus:    
        assert corpus is not None, "Corpus missing!"

    model.eval()
    device = next(model.parameters()).device
    label_ids_list: list[torch.LongTensor] = []
    prediction_ids_list: list[torch.LongTensor] = []
    doc_ids_list: list[torch.IntTensor] = []
    attn_mask_list: list[torch.FloatTensor] = []
    n_batches = len(dataloader)
    eval_start_time = time.time()
    logging.info(f"Inference batch size = {batch_size}")
    logging.info(f"Number of inference batches = {n_batches}")
    logging.info("Starting inference...")

    with torch.no_grad():
        running_batch_times = []
        for i, batch in enumerate(dataloader):
            (batch_token_ids, batch_mention_ids, batch_label_ids,
                batch_attn_mask, batch_global_attn_mask, batch_doc_ids) = batch

            if use_dynamic_loading:
                batch_token_ids = batch_token_ids.to(device)
                batch_mention_ids = batch_mention_ids.to(device)
                batch_attn_mask = batch_attn_mask.to(device)
                batch_global_attn_mask = batch_global_attn_mask.to(device)

            start_time = time.time()
            batch_logits = model(batch_token_ids,
                batch_mention_ids, batch_attn_mask, batch_global_attn_mask)
            batch_prediction_ids = batch_logits.argmax(dim=2)
            label_ids_list.append(batch_label_ids.cpu())
            prediction_ids_list.append(batch_prediction_ids.detach().cpu())
            doc_ids_list.append(batch_doc_ids.cpu())
            attn_mask_list.append(batch_attn_mask.cpu())
            time_taken = time.time() - start_time
            running_batch_times.append(time_taken)

            if (i + 1) % print_n_batches == 0:
                average_time_per_batch = np.mean(running_batch_times)
                estimated_time_remaining = (n_batches - i - 1) * (
                    average_time_per_batch)
                average_time_per_batch_str = (
                    util.convert_float_seconds_to_time_string(
                        average_time_per_batch))
                estimated_time_remaining_str = (
                    util.convert_float_seconds_to_time_string(
                        estimated_time_remaining))
                time_elapsed_str = util.convert_float_seconds_to_time_string(
                    time.time() - eval_start_time)
                running_batch_times = []

                logging.info(f"Batch {i + 1}")
                logging.info(f"Time elapsed in inference = {time_elapsed_str}")
                logging.info("Average inference time @ batch = "
                        f"{average_time_per_batch_str}")
                logging.info("Estimated inference time remaining = "
                        f"{estimated_time_remaining_str}")

    time_taken = time.time() - eval_start_time
    time_taken_str = util.convert_float_seconds_to_time_string(time_taken)
    logging.info("...Inference done.")
    logging.info(f"Total time taken in inference = {time_taken_str}")

    groundtruth = torch.cat(label_ids_list, dim=0)
    predictions = torch.cat(prediction_ids_list, dim=0)
    doc_ids = torch.cat(doc_ids_list, dim=0)
    attn_mask = torch.cat(attn_mask_list, dim=0)
    start_time = time.time()
    if use_official:
        logging.info("Starting official evaluation...")
        output = evaluate_tensors_official(official_scorer, groundtruth,
                    predictions, doc_ids, attn_mask, corpus,
                    evaluate_against_corpus)
    else:
        logging.info("Starting scorch evaluation...")
        output = evaluate_tensors_scorch(groundtruth, predictions, doc_ids,
                    attn_mask, corpus, evaluate_against_corpus)
    logging.info("...Evaluation done.")
    time_taken = time.time() - start_time
    time_taken_str = util.convert_float_seconds_to_time_string(time_taken)
    logging.info(f"Time taken in evaluation = {time_taken_str}")

    time_taken = time.time() - eval_start_time
    time_taken_str = util.convert_float_seconds_to_time_string(time_taken)
    logging.info("Total time taken in inference and evaluation = "
            f"{time_taken_str}")
    return output