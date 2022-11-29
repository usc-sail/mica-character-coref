"""Baselines for movie coreference resolution. Run a pre-trained neural coreference resolution
model on the scripts.
"""
# pyright: reportGeneralTypeIssues=false

from mica_text_coref.coref.word_level_coref.coref import CorefModel
from mica_text_coref.coref.word_level_coref.coref.tokenizer_customization import TOKENIZER_FILTERS, TOKENIZER_MAPS
from mica_text_coref.coref.movie_coref.data import CorefCorpus
from mica_text_coref.coref.movie_coref import conll
from mica_text_coref.coref.movie_coref import evaluate
from mica_text_coref.coref.movie_coref.result import Metric
from mica_text_coref.coref.movie_coref import rules
from mica_text_coref.coref.movie_coref import split_and_merge

from collections import defaultdict
import jsonlines
import os
import re
import torch
import tqdm

class GraphNode:
    def __init__(self, node_id: int):
        self.id = node_id
        self.links: set[GraphNode] = set()
        self.visited = False

    def link(self, another: "GraphNode"):
        self.links.add(another)
        another.links.add(self)

    def __repr__(self) -> str:
        return str(self.id)

def wl_build_doc(doc: dict, model: CorefModel) -> dict:
    filter_func = TOKENIZER_FILTERS.get(model.config.bert_model, lambda _: True)
    token_map = TOKENIZER_MAPS.get(model.config.bert_model, {})
    word2subword = []
    subwords = []
    word_id = []
    for i, word in enumerate(doc["cased_words"]):
        tokenized_word = (token_map[word] if word in token_map else model.tokenizer.tokenize(word))
        tokenized_word = list(filter(filter_func, tokenized_word))
        word2subword.append((len(subwords), len(subwords) + len(tokenized_word)))
        subwords.extend(tokenized_word)
        word_id.extend([i] * len(tokenized_word))
    doc["word2subword"] = word2subword
    doc["subwords"] = subwords
    doc["word_id"] = word_id
    doc["head2span"] = []
    if "speaker" not in doc: doc["speaker"] = ["_" for _ in doc["cased_words"]]
    doc["word_clusters"] = []
    doc["span_clusters"] = []
    return doc

def wl_predict(config_file: str, weights: str, batch_size: int, genre: str, input_file: str, output_file: str, split_len: int | None, overlap_len: int, use_gpu: bool):
    """Predict coreference clusters using the word-level coreference model. Save predictions to output file."""
    # Initialize model
    use_gpu = use_gpu and torch.cuda.is_available()
    model = CorefModel(config_file, "roberta", use_gpu = use_gpu)
    model.config.a_scoring_batch_size = batch_size
    model.load_weights(path=weights, map_location=model.config.device, ignore={"bert_optimizer", "general_optimizer", "bert_scheduler", "general_scheduler"})
    model.training = False

    # Collect docs and split them (optional)
    docs = []
    with jsonlines.open(input_file, mode="r") as input_data:
        for doc in input_data:
            if split_len is None:
                docs.append(doc)
            else:
                for small_doc in split_and_merge.split_screenplay(doc, split_len, overlap_len, verbose=True):
                    docs.append(small_doc)

    # Inference
    results = {}
    with torch.no_grad():
        tbar = tqdm.tqdm(docs, unit="docs")
        for doc in tbar:
            tbar.set_description(doc["document_id"])
            doc["document_id"] = genre + "_" + doc["document_id"]
            doc = wl_build_doc(doc, model)
            result = model.run(doc)
            results[doc["document_id"]] = result.__dict__
            doc["span_clusters"] = result.span_clusters
            doc["word_clusters"] = result.word_clusters
            for key in ("word2subword", "subwords", "word_id", "head2span"): del doc[key]

    # Write output docs
    with jsonlines.open(output_file + ".jsonlines", mode="w") as output_data:
        output_data.write_all(docs)

    # Write result tensor dictionary
    torch.save(results, output_file + ".pt")

def get_scores_indices_heads(pt: dict, offset: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor, dict[int, tuple[int, int]]]:
    coref, ind, word_clusters, span_clusters = pt["coref_scores"], pt["top_indices"], pt["word_clusters"], pt["span_clusters"]
    ind = ind + offset[0]
    heads = [word + offset[0] for cluster in word_clusters for word in cluster]
    spans = [(p + offset[0], q + offset[0]) for cluster in span_clusters for p, q in cluster]
    head2span = {head: span for head, span in zip(heads, spans)}
    return coref, ind, head2span

def clusterize(scores: torch.Tensor, top_indices: torch.Tensor) -> list[list[int]]:
    antecedents = scores.argmax(dim=1) - 1
    not_dummy = antecedents >= 0
    coref_span_heads = torch.arange(0, len(scores))[not_dummy]
    antecedents = top_indices[coref_span_heads, antecedents[not_dummy]]

    nodes = [GraphNode(i) for i in range(len(scores))]
    for i, j in zip(coref_span_heads.tolist(), antecedents.tolist()):
        nodes[i].link(nodes[j])
        assert nodes[i] is not nodes[j]

    clusters: list[list[int]] = []
    for node in nodes:
        if len(node.links) > 0 and not node.visited:
            cluster: list[int] = []
            stack = [node]
            while stack:
                current_node = stack.pop()
                current_node.visited = True
                cluster.append(current_node.id)
                stack.extend(link for link in current_node.links if not link.visited)
            assert len(cluster) > 1
            clusters.append(sorted(cluster))
    return sorted(clusters)

def wl_evaluate(reference_scorer: str, config_file: str, weights: str, batch_size: int, genre: str, input_file: str, output_file: str, entity: str, merge_speakers: bool, provide_gold_mentions: bool, 
    remove_gold_singletons: bool, split_len: int | None, overlap_len: int, merge_strategy: str, use_reference_scorer: bool, calc_results: bool, 
    overwrite: bool, use_gpu: bool) -> dict[str, dict[str, Metric]] | None:
    """Evaluate coreference using word-level coreference model.

    Args:
        reference_scorer (str): Path to conll perl reference scorer.
        config_file (str): Config file used by the word-level roberta coreference model.
        weights (str): Path to the weights of the word-level roberta coreference model.
        batch_size (int): Batch size to use by the word-level roberta coreference model for antecedent scoring.
        genre (str): Genre to use by word-level roberta coreference model.
        input_file (str): Jsonlines file containing gold annotations and screenplay.
        output_file (str): Output file to which the predictions (jsonlines and tensors) will be saved.
        entity (str): Type of entity to keep.
        merge_speakers (bool): If true, merge clusters by speakers.
        provide_gold_mentions (bool): If true, provide gold mentions to the model.
        remove_gold_singletons (bool): If true, remove gold clusters with only a single mention.
        split_len (int|None): Number of words of the smaller screenplays. If None, no splitting occurs.
        overlap_len (int): Overlap in words between smaller screenplays.
        merge_strategy (str): Method to merge predictions, can be "none", "before", "after", "average", "max", or "min"
        use_reference_scorer (bool): If true, use reference scorer.
        calc_results (bool): If true, calculate results.
        overwrite (bool): If true, run prediction even if output file is present.
        use_gpu (bool): If true, use cuda:0 gpu if available.
    
    Returns:
        Dictionary of coreference scores keyed by metric and movie name. The keys of the outer dictionary are metric names: muc, bcub, and ceafe. The keys of the inner dictionary are movie names and
        a special key called "all" that contains micro-averaged scores. The values are Metric objects.
    """
    # Run inference and write predictions to output file
    docs_file = output_file + ".jsonlines"
    pt_file = output_file + ".pt"
    if overwrite or not os.path.exists(docs_file) or not os.path.exists(pt_file):
        wl_predict(config_file, weights, batch_size, genre, input_file, output_file, split_len, overlap_len, use_gpu)
    
    # Return if calc_results is False
    if not calc_results:
        return

    gold_lines = []
    pred_lines = []
    movie_to_gold_clusters: dict[str, list[set[tuple[int, int]]]] = {}
    movie_to_pred_clusters: dict[str, list[set[tuple[int, int]]]] = {}

    # Read predictions
    device = "cuda:0" if use_gpu else "cpu"
    with jsonlines.open(output_file + ".jsonlines") as reader:
        pred_docs = {doc["document_id"]: doc for doc in reader}
    pt = torch.load(output_file + ".pt", map_location=device)
    corpus = CorefCorpus(input_file)
    gold_docs = {doc.movie: doc for doc in corpus}

    # Movie to number of parts
    movie_to_n_parts = defaultdict(int)
    for doc_id in pred_docs.keys():
        match = re.match(r"[a-z]{2}_(\w+)_(\d+)", doc_id)
        assert match is not None, "Improperly formatted document id"
        movie = match.group(1)
        part = int(match.group(2))
        movie_to_n_parts[movie] = max(part, movie_to_n_parts[movie])
    
    # Loop over movie and parts
    for movie, n_parts in movie_to_n_parts.items():
        # Merge predictions
        corefs, inds, offsets, head2span = [], [], [], {}
        for i in range(1, n_parts + 1):
            offset = pred_docs[f"{genre}_{movie}_{i}"]["offset"]
            coref, ind, _head2span = get_scores_indices_heads(pt[f"{genre}_{movie}_{i}"], offset)
            corefs.append(coref)
            inds.append(ind)
            offsets.append(offset)
            head2span.update(_head2span)
        overlap_lens = [offsets[i][1] - offsets[i + 1][0] for i in range(n_parts - 1)]
        coref_lens = [len(coref) for coref in corefs]
        coref, ind = split_and_merge.combine_coref_scores(corefs, inds, overlap_lens, merge_strategy)

        # Get predicted clusters
        word_clusters = clusterize(coref, ind)
        span_clusters = []
        for cluster in word_clusters:
            span_cluster = []
            for head in cluster:
                if head in head2span:
                    span_cluster.append(head2span[head])
            if span_cluster:
                span_clusters.append(span_cluster)
        n_word_mentions = sum([len(cluster) for cluster in word_clusters])
        n_span_mentions = sum([len(cluster) for cluster in span_clusters])

        gold_doc = gold_docs[movie]
        gold_clusters = [set([(mention.begin, mention.end) for mention in mentions]) for mentions in gold_doc.clusters.values()]
        pred_clusters = [set([(i, j - 1) for i, j in cluster]) for cluster in span_clusters]

        # Merge predicted clusters by speaker names
        if merge_speakers:
            pred_clusters = rules.merge_speakers(gold_doc.token, gold_doc.parse, pred_clusters)

        # Filter predicted clusters by entity type
        if entity == "speaker":
            pred_clusters = rules.keep_speakers(gold_doc.parse, pred_clusters)
        elif entity == "person":
            pred_clusters = rules.keep_persons(gold_doc.ner, pred_clusters)

        # Remove gold clusters containing single mention
        if remove_gold_singletons:
            gold_clusters = rules.remove_singleton_clusters(gold_clusters)

        # Filter predicted mentions by gold mentions
        if provide_gold_mentions:
            gold_mentions = set([mention for cluster in gold_clusters for mention in cluster])
            pred_clusters = rules.filter_mentions(gold_mentions, pred_clusters)
        
        gold_lines.extend(conll.convert_to_conll(gold_doc, gold_clusters))
        pred_lines.extend(conll.convert_to_conll(gold_doc, pred_clusters))
        movie_to_gold_clusters[movie] = gold_clusters
        movie_to_pred_clusters[movie] = pred_clusters

    # Evaluate using perl scorer
    gold_file = os.path.join(os.path.dirname(os.path.normpath(output_file)), "gold.conll")
    pred_file = os.path.join(os.path.dirname(os.path.normpath(output_file)), "pred.conll")
    if use_reference_scorer:
        _result = conll.evaluate_conll(reference_scorer, gold_lines, pred_lines, gold_file, pred_file)
    else:
        _result = evaluate.evaluate(movie_to_gold_clusters, movie_to_pred_clusters)
    result = defaultdict(lambda: defaultdict(lambda: Metric))

    # Convert into Metric objects
    for metric, metric_result in _result.items():
        for movie, movie_result in metric_result.items():
            result[metric][movie] = Metric(*movie_result)
    result = dict(result)
    return result