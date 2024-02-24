"""Baselines for movie coreference resolution. Run a pre-trained neural coreference resolution model on the scripts."""
import conll
import data
import evaluate
import rules
import split_and_merge
from ..coref.word_level_coref.coref import CorefModel
from ..coref.word_level_coref.coref import tokenizer_customization

import collections
import copy
import itertools
import jsonlines
import os
import re
import torch
import tqdm


class GraphNode:
    """Word node in forest of word clusters"""
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
    """Tokenize words"""
    filter_func = tokenizer_customization.TOKENIZER_FILTERS.get(model.config.bert_model, lambda _: True)
    token_map = tokenizer_customization.TOKENIZER_MAPS.get(model.config.bert_model, {})
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

def wl_predict(config_file: str, weights: str, batch_size: int, genre: str, input_file: str, output_file: str,
               split_len: int | None, overlap_len: int):
    """Predict coreference clusters using the word-level coreference model. Save predictions to output file.

    Args:
        config_file: Filepath of the TOML configuration file.
        weights: Filepath of the trained weights.
        batch_size: Batch size used for the fine antecedent scoring.
        genre: Genre of the documents.
        input_file: Jsonlines file of the documents for which we run the model,
        output_file: File name of the jsonlines and tensor file which is written.
        split_len: Size of the subdocuments in words, if None then no splitting occurs.
        overlap_len: Size of the overlap between successive subdocuments in words.
    """
    # Initialize model
    # Pass build_optimizers=False because we only use the model for inference
    model = CorefModel(config_file, "roberta", build_optimizers=False)
    model.config.a_scoring_batch_size = batch_size
    model.load_weights(path=weights, map_location=model.config.device,
                       ignore={"bert_optimizer", "general_optimizer", "bert_scheduler", "general_scheduler"})
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

def get_scores_indices_heads(pt: dict, offset: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor,
                                                                         dict[int, tuple[int, int]]]:
    """Get the list of coreference scores, top-scoring antecedent indices, and offsets from the tensor."""
    coref, ind, word_clusters, span_clusters = (pt["coref_scores"], pt["top_indices"], pt["word_clusters"],
                                                pt["span_clusters"])
    ind = ind + offset[0]
    heads = [word + offset[0] for cluster in word_clusters for word in cluster]
    spans = [(p + offset[0], q + offset[0]) for cluster in span_clusters for p, q in cluster]
    head2span = {head: span for head, span in zip(heads, spans)}
    return coref, ind, head2span

def clusterize(scores: torch.Tensor, top_indices: torch.Tensor) -> list[list[int]]:
    """Find the clusters from the coreference scores and top indices"""
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

def wl_evaluate(input_file: str,
                output_file: str,
                reference_scorer: str,
                config_file: str,
                weights: str,
                batch_size: int,
                preprocess: str,
                genre: str,
                split_len: int,
                overlap_len: int,
                use_reference_scorer: bool,
                calc_results: bool,
                overwrite: bool,
                lea: bool) -> (dict[tuple[str, bool, str, bool, bool],
                                    dict[str, dict[str, data.Metric]]] | None):
    """Evaluate coreference using word-level coreference model.

    Args:
        input_file (str): Jsonlines file containing gold annotations and screenplay.
        output_file (str): Output file to which the predictions (jsonlines and tensors) will be saved.
        reference_scorer (str): Path to conll perl reference scorer.
        config_file (str): Config file used by the word-level roberta coreference model.
        weights (str): Path to the weights of the word-level roberta coreference model.
        batch_size (int): Batch size to use by the word-level roberta coreference model for antecedent scoring.
        preprocess (str): Preprocessing used on the input file
        genre (str): Genre to use by word-level roberta coreference model.
        split_len (int): Number of words of the smaller screenplays.
        overlap_len (int): Overlap in words between smaller screenplays.
        use_reference_scorer (bool): If true, use reference scorer.
        calc_results (bool): If true, calculate results.
        overwrite (bool): If true, run prediction even if output file is present.
        lea (bool): If true, evaluate link-based entity aware metric
    
    Returns:
        Dictionary of coreference scores in the following hierarchy: setting/movie/metric
            setting (tuple): (merge strategy (str), merge speakers (bool), entity (str), remove_gold_singletons (bool), 
                              provide_gold_mentions (bool))
            movie (str): movie name or "micro"
            metric (str): muc, bcub, ceafe, lea

    Return None if calc_results is false        
    """
    evaluator = evaluate.Evaluator()

    # Run inference and write predictions to output file
    # The jsonlines file contains model predictions
    # The pt file contains model scores for each subdocument used for fusion
    docs_file = output_file + ".jsonlines"
    pt_file = output_file + ".pt"
    if overwrite or not os.path.exists(docs_file) or not os.path.exists(pt_file):
        wl_predict(config_file, weights, batch_size, genre, input_file, output_file, split_len, overlap_len)
    
    # Return if calc_results is False
    if not calc_results:
        return
    
    # Define the domain of each setting: merge strategy, merge speakers, entities, remove gold singletons, and 
    # provide gold mentions
    # settings iterates over the cartesian product (all possible permutation) of each setting's domain
    merge_strategy_arr = ["pre", "post", "avg", "max", "min", "none"]
    merge_speakers_arr = [False, True]
    entity_arr = ["all", "speaker", "person"]
    remove_gold_singletons_arr = [False, True]
    provide_gold_mentions_arr = [False, True]
    settings = list(itertools.product(merge_speakers_arr, entity_arr, remove_gold_singletons_arr,
                                      provide_gold_mentions_arr))
    n_settings = 2 * 3 * 2 * 2

    # Initialize variables required for evaluation
    # settings_to_movie_to_gold_clusters is a dictionary from setting to movie name to list of gold clusters
    # settings_to_movie_to_pred_clusters is a dictionary from setting to movie name to list of predicted clusters
    # setting_to_gold_lines is a dictionary from setting to conll lines containing gold annotations of all movies
    # setting_to_pred_lines is a dictionary from setting to conll lines containing predicted annotations of all movies
    setting_to_movie_to_gold_clusters = collections.defaultdict(lambda: collections.defaultdict())
    setting_to_movie_to_pred_clusters = collections.defaultdict(lambda: collections.defaultdict())
    setting_to_gold_lines = collections.defaultdict(list)
    setting_to_pred_lines = collections.defaultdict(list)
    result: dict[tuple[str, bool, str, bool, bool], dict[str, dict[str, data.Metric]]] = {}

    # Read predictions into pred_docs.
    # pred_docs is a dictionary of document_id (<movie_name>_<part_id>) to model predictions
    # pt is also dictionary from document_id (<movie_name>_<part_id>) to model scores
    # corpus is a CorefCorpus object containing CorefDocument objects for each movie
    # gold_docs is a dictionary of movie_name to the movie's CorefDocument object
    with jsonlines.open(output_file + ".jsonlines") as reader:
        pred_docs = {doc["document_id"]: doc for doc in reader}
    pt = torch.load(output_file + ".pt", map_location="cpu")
    corpus = data.CorefCorpus(input_file)
    gold_docs = {doc.movie: doc for doc in corpus}

    # Find the number of parts of each movie
    # movie_to_n_parts is a dictionary from movie_name to the number of subdocuments (parts) the movie screenplay was
    # split into
    movie_to_n_parts = collections.defaultdict(int)
    for doc_id in pred_docs.keys():
        match = re.match(r"[a-z]{2}_(\w+)_(\d+)", doc_id)
        assert match is not None, "Improperly formatted document id"
        movie = match.group(1)
        part = int(match.group(2))
        movie_to_n_parts[movie] = max(part, movie_to_n_parts[movie])
    
    # Loop over movie and parts
    for movie, n_parts in tqdm.tqdm(movie_to_n_parts.items(), unit="movie", position=0):

        # Merge predictions
        # corefs is a list of tensors of the coreference scores from each subdocument
        # inds is a list of top-scoring antecedents from each subdocument
        # offsets is a list of ordered integer pairs denoting the subdocument offset in the original screenplay
        # head2span is a dictionary from head id to an ordered pair of integers denoting its expanded text span
        # corefs and inds are merged into a single tensor for coreference scores and top-scoring antecedents
        # denoted by coref and ind
        corefs, inds, offsets, head2span = [], [], [], {}
        for i in range(1, n_parts + 1):
            offset = pred_docs[f"{genre}_{movie}_{i}"]["offset"]
            coref, ind, _head2span = get_scores_indices_heads(pt[f"{genre}_{movie}_{i}"], offset)
            corefs.append(coref)
            inds.append(ind)
            offsets.append(offset)
            head2span.update(_head2span)
        overlap_lens = [offsets[i][1] - offsets[i + 1][0] for i in range(n_parts - 1)]

        # Loop over merge strategies
        for merge_strategy in tqdm.tqdm(merge_strategy_arr, unit="strategy", position=1, leave=False):
            coref, ind = split_and_merge.combine_coref_scores(corefs, inds, overlap_lens, merge_strategy)

            # Find the predicted word clusters from the merged coreference scores and top-scoring antecedents
            # Derive the predicted span clusters from the predicted word clusters using head2span dictionary
            word_clusters = clusterize(coref, ind)
            span_clusters = []
            for cluster in word_clusters:
                span_cluster = []
                for head in cluster:
                    if head in head2span:
                        span_cluster.append(head2span[head])
                if span_cluster:
                    span_clusters.append(span_cluster)

            # Find the gold_clusters from gold_docs and pred_clusters from the predicted span clusters
            # Each cluster is a set of text span indices (inclusive end)
            gold_doc = gold_docs[movie]
            gold_clusters_ = [set([(mention.begin, mention.end) for mention in mentions])
                                for mentions in gold_doc.clusters.values()]
            pred_clusters_ = [set([(i, j - 1) for i, j in cluster]) for cluster in span_clusters]

            # Loop over settings
            for merge_speakers, entity, remove_gold_singletons, provide_gold_mentions in tqdm.tqdm(
                    settings, unit="setting", total=n_settings, position=2, leave=False):
                setting = (merge_strategy, merge_speakers, entity, remove_gold_singletons, provide_gold_mentions)
                gold_clusters = copy.deepcopy(gold_clusters_)
                pred_clusters = copy.deepcopy(pred_clusters_)

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
                
                # If preprocess == "addsays" or "none", remove spans from gold and pred clusters that
                # overlap with a speaker. We perform this step so that it is a fair comparison between
                # preprocess == "nocharacters" and preprocess == "addsays" or "none"
                if preprocess == "addsays" or preprocess == "none":
                    gold_clusters = rules.remove_speaker_links(gold_clusters, gold_doc.parse)
                    pred_clusters = rules.remove_speaker_links(pred_clusters, gold_doc.parse)
                
                # Record the gold and predicted clusters, and their conll formats to the setting dictionaries
                setting_to_movie_to_gold_clusters[setting][movie] = gold_clusters
                setting_to_movie_to_pred_clusters[setting][movie] = pred_clusters
                if not lea:
                    setting_to_gold_lines[setting].extend(conll.convert_to_conll(gold_doc, gold_clusters))
                    setting_to_pred_lines[setting].extend(conll.convert_to_conll(gold_doc, pred_clusters))

    for setting in tqdm.tqdm(setting_to_movie_to_pred_clusters.keys(), unit="setting"):
        movie_to_gold_clusters = setting_to_movie_to_gold_clusters[setting]
        movie_to_pred_clusters = setting_to_movie_to_pred_clusters[setting]
        if lea:
            result[setting] = evaluator.pycoref(movie_to_gold_clusters, movie_to_pred_clusters, metrics=["lea"])
        else:
            gold_lines = setting_to_gold_lines[setting]
            pred_lines = setting_to_pred_lines[setting]
            gold_file = os.path.join(os.path.dirname(os.path.normpath(output_file)), "gold.conll")
            pred_file = os.path.join(os.path.dirname(os.path.normpath(output_file)), "pred.conll")
            if use_reference_scorer:
                with open(gold_file, "w") as fw:
                    fw.writelines(gold_lines)
                with open(pred_file, "w") as fw:
                    fw.writelines(pred_lines)
                result[setting] = evaluator.perlcoref(gold_file, pred_file, reference_scorer)
                os.remove(gold_file)
                os.remove(pred_file)
            else:
                result[setting] = evaluator.pycoref(movie_to_gold_clusters, movie_to_pred_clusters,
                                                     metrics=["muc", "bcub", "ceafe"])
    
    return result