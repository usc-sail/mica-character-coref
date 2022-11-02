"""Baselines for movie coreference resolution. Run a pre-trained neural coreference resolution
model on the scripts.
"""

from mica_text_coref.coref.word_level_coref.coref import CorefModel
from mica_text_coref.coref.word_level_coref.coref.tokenizer_customization import *
from mica_text_coref.coref.movie_coref.data import CorefCorpus
from mica_text_coref.coref.movie_coref import conll
from mica_text_coref.coref.movie_coref.result import Metric
from mica_text_coref.coref.movie_coref import rules

import jsonlines
import os
import torch
import tqdm

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
    if "speaker" not in doc:
        doc["speaker"] = ["_" for _ in doc["cased_words"]]
    doc["word_clusters"] = []
    doc["span_clusters"] = []
    return doc

def wl_predict(config_file: str, weights: str, batch_size: int, genre: str, input_file: str,
    output_file: str):
    """Predict coreference clusters using the word-level coreference model. Save predictions to 
    output file.
    """
    model = CorefModel(config_file, "roberta")
    model.config.a_scoring_batch_size = batch_size
    model.load_weights(path=weights, map_location="cpu", ignore={"bert_optimizer", 
        "general_optimizer", "bert_scheduler", "general_scheduler"})
    model.training = False
    with jsonlines.open(input_file, mode="r") as input_data:
        docs = [wl_build_doc(doc, model) for doc in input_data]
    with torch.no_grad():
        for doc in tqdm.tqdm(docs, unit="docs"):
            doc["document_id"] = genre + "_" + doc["document_id"]
            result = model.run(doc)
            doc["span_clusters"] = result.span_clusters
            doc["word_clusters"] = result.word_clusters
            for key in ("word2subword", "subwords", "word_id", "head2span"):
                del doc[key]
    with jsonlines.open(output_file, mode="w") as output_data:
        output_data.write_all(docs)

def wl_evaluate(reference_scorer: str, config_file: str, weights: str, batch_size: int, genre: str,
    input_file: str, output_file: str, entity: str, merge_speakers: bool, 
    provide_gold_mentions: bool, remove_gold_singletons: bool, overwrite: bool
    ) -> tuple[Metric, Metric, Metric, float]:
    """Evaluate coreference using word-level coreference model.

    Args:
        reference_scorer (str): Path to conll perl reference scorer.
        config_file (str): Config file used by the word-level roberta coreference model.
        weights (str): Path to the weights of the word-level roberta coreference model.
        batch_size (int): Batch size to use by the word-level roberta coreference model for antecedent
            scoring.
        genre (str): Genre to use by word-level roberta coreference model.
        input_file (str): Jsonlines file containing gold annotations and screenplay.
        output_file (str): Jsonlines file to which the predictions will be saved.
        entity (str): Type of entity to keep.
        merge_speakers (bool): If true, merge clusters by speakers.
        provide_gold_mentions (bool): If true, provide gold mentions to the model.
        remove_gold_singletons (bool): If true, remove gold clusters with only a single mention.
        overwrite (bool): If true, run prediction even if output file is present.
    
    Returns:
        tuple [Metric, Metric, Metric, float]: MUC, B_cubed, and CEAFe metric, and the average F1.
    """
    if overwrite or not os.path.exists(output_file):
        wl_predict(config_file, weights, batch_size, genre, input_file, output_file)

    docid_to_output = {}
    with jsonlines.open(output_file) as reader:
        for doc in reader:
            docid_to_output[doc["document_id"]] = doc

    corpus = CorefCorpus(input_file)
    gold_lines, pred_lines = [], []
    for document in corpus:
        key = f"{genre}_{document.movie}"
        output_doc = docid_to_output[key]
        gold_clusters = [set([(mention.begin, mention.end) for mention in mentions]) 
            for mentions in document.clusters.values()]
        pred_clusters = [set([(i, j - 1) for i, j in cluster]) 
            for cluster in output_doc["span_clusters"]]

        # Merge predicted clusters by speaker names
        if merge_speakers:
            pred_clusters = rules.merge_speakers(document.token, document.parse, pred_clusters)

        # Filter predicted clusters by entity type
        if entity == "speaker":
            pred_clusters = rules.keep_speakers(document.parse, pred_clusters)
        elif entity == "person":
            pred_clusters = rules.keep_persons(document.ner, pred_clusters)

        # Remove gold clusters containing single mention
        if remove_gold_singletons:
            gold_clusters = rules.remove_singleton_clusters(gold_clusters)

        # Filter predicted mentions by gold mentions
        if provide_gold_mentions:
            gold_mentions = set([mention for cluster in gold_clusters for mention in cluster])
            pred_clusters = rules.filter_mentions(gold_mentions, pred_clusters)

        gold_lines.extend(conll.convert_to_conll(document, gold_clusters))
        pred_lines.extend(conll.convert_to_conll(document, pred_clusters))

    gold_file = os.path.join(os.path.dirname(os.path.normpath(output_file)), "gold.conll")
    pred_file = os.path.join(os.path.dirname(os.path.normpath(output_file)), "pred.conll")
    values = conll.evaluate_conll(reference_scorer, gold_lines, pred_lines, gold_file, pred_file)
    os.remove(gold_file)
    os.remove(pred_file)
    muc_metric = Metric(*values[:2])
    b_cubed_metric = Metric(*values[2:4])
    ceafe_metric = Metric(*values[4:])
    average_f1 = (muc_metric.f1 + b_cubed_metric.f1 + ceafe_metric.f1)/3
    print(f"MUC = {muc_metric}")
    print(f"B_cubed = {b_cubed_metric}")
    print(f"CEAFe = {ceafe_metric}")
    print(f"Average F1 = {average_f1:.4f}")
    return muc_metric, b_cubed_metric, ceafe_metric, average_f1