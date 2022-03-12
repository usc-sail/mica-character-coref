import argparse
from collections import defaultdict
import spacy
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from evaluate_coreference import evaluate_coreference
from allennlp.predictors.predictor import Predictor
from heuristic_coreference_resolution import heuristic_coreference_resolution
import random

def merge_clusters(clusters):
    
    def find_mergeable_clusters(clusters):
        for i in range(len(clusters)):
            for j in range(len(clusters)):
                if i != j and len(clusters[i].intersection(clusters[j])) > 0:
                    return i, j
        return -1, -1
        
    while True:
        i, j = find_mergeable_clusters(clusters)
        if i != -1:
            clusters = [clusters[k] for k in range(len(clusters)) if k != i and k != j] + [clusters[i].union(clusters[j])]
        else:
            break
        
    return clusters

def evaluate_coreference_by_joining_elements(parsed_file, coref_file, random_seed=None, cuda_device=-1, keep_only_speaker_gold_clusters=False, remove_singleton_gold_clusters=False, use_speaker_sep=False, keep_person_sys_clusters=False, keep_speaker_sys_clusters=False, heuristic_speaker_resolution=False, heuristic_pronoun_resolution=False, min_speaker_sim=0.6, max_speaker_merges=3, coreference_model=None, cache_document_to_coref_result={}):
    print("loading spacy model")
    spacy_pipeline = spacy.load("en_core_web_sm")
    
    print("spacy tokenization of screenplay elements")
    tags, elements, spacy_docs, count_spacy_docs, mention_tags = [], [], [], [], []

    lines = open(parsed_file).read().strip().split("\n")
    
    start_line_index = 0
    end_line_index = len(lines)
    if random_seed is not None:
        random.seed(random_seed)
        start_line_index = random.randint(0, 250)
        while lines[start_line_index][0] not in ["S", "N", "C"]:
            start_line_index = random.randint(0, 250)
        end_line_index = start_line_index + random.randint(200, 300)
        end_line_index = max(len(lines), end_line_index)
        print(f"start line index = {start_line_index}, end line index = {end_line_index}")

    for line_index in trange(start_line_index, end_line_index):
        line = lines[line_index]
        tag, element = line[0], line[2:].strip()
        spacy_doc = spacy_pipeline(element)
        tags.append(tag)
        elements.append(element)
        spacy_docs.append(spacy_doc)
        count_spacy_docs.append(len(spacy_doc))
        mention_tags.extend([tag] * len(spacy_doc) + ["X"])
    mention_tags = mention_tags[:-1]

    print("finding global gold mention positions")
    coref_df = pd.read_csv(coref_file, index_col=None)

    if random_seed is not None:
        print(f"choosing subset of gold mentions between {start_line_index} and {end_line_index}")
        coref_df = coref_df[(coref_df.pbegin_ind >= start_line_index) and (coref_df.end_line_index < end_line_index)]

    mi = np.full(len(coref_df), np.nan, dtype=np.float)
    mj = np.full(len(coref_df), np.nan, dtype=np.float)

    for ind, row in coref_df.iterrows():
        if row.pbegin_ind == row.pend_ind:
            i, j, k = int(row.pbegin_ind), int(row.pbegin_pos), int(row.pend_pos)
            offset = sum(count_spacy_docs[:i]) + i
            for token in spacy_docs[i]:
                if token.idx == j:
                    mi[ind] = token.i + offset
                if token.idx + len(token) == k + 1:
                    mj[ind] = token.i + offset

    coref_df["mention_start"] = mi
    coref_df["mention_end"] = mj
    
    n = len(coref_df)
    m = (coref_df.pbegin_ind == coref_df.pend_ind).sum()
    p = (coref_df.mention_start.notna() & coref_df.mention_end.notna()).sum()
    print(f"\t{n} gold mentions")
    print(f"\t{m} ({100*m/n:.2f}%) gold mentions found after parse")
    print(f"\t{p} ({100*p/n:.2f}%) gold mentions' spacy tokenization span found")

    print("finding gold clusters")
    coref_df = coref_df[coref_df.mention_start.notna() & coref_df.mention_end.notna()].astype({"mention_start": int, "mention_end": int})

    if keep_only_speaker_gold_clusters:
        print("\tkeeping only speaker gold clusters")
        coref_df = coref_df[coref_df.SPEAKER]

    gold_clusters = defaultdict(set)
    for entity, df in coref_df.groupby("entityLabel"):
        gold_cluster = set()
        for _, row in df.iterrows():
            gold_cluster.add((row.mention_start, row.mention_end))
        gold_clusters[entity] = gold_cluster

    if remove_singleton_gold_clusters:
        print("\tremoving singleton gold clusters")
        for entity, cluster in gold_clusters.copy().items():
            if len(cluster) == 1:
                gold_clusters.pop(entity)

    print(f"{len(gold_clusters)} gold clusters")

    if coreference_model is None:
        print("loading allennlp coreference model")
        predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz", cuda_device=cuda_device)
    else:
        predictor = coreference_model

    print("finding sys clusters")
    if use_speaker_sep:
        print("\tusing 'says' after character names")
        document = ""
        for element, tag in zip(elements, tags):
            if tag == "C":
                document += element + " says "
            else:
                document += element + "\n"
        document = document.strip()
    else:
        document = "\n".join(elements)

    if cache_document_to_coref_result is not None and document in cache_document_to_coref_result:
        coref_result = cache_document_to_coref_result[document]
    else:
        print("\tallennlp coreference resolution")
        coref_result = predictor.predict(document=document)

    print("\tspacy ner on document")
    spacy_document = spacy_pipeline(document)

    if keep_speaker_sys_clusters:
        print("\tkeeping speaker sys clusters")

    if keep_person_sys_clusters:
        print("\tkeeping person sys clusters")

    sys_clusters = []
    for cluster in coref_result["clusters"]:
        sys_cluster = set()
        is_speaker = False
        is_person = False

        for i, j in cluster:
            is_speaker |= any([mention_tags[k] == "C" for k in range(i, j + 1)])
            is_person |= any([spacy_document[k].ent_type_ == "PERSON" for k in range(i, j + 1)])
            sys_cluster.add((i, j))

        w = not keep_speaker_sys_clusters and not keep_person_sys_clusters
        x = keep_speaker_sys_clusters and not keep_person_sys_clusters and is_speaker
        y = not keep_speaker_sys_clusters and keep_person_sys_clusters and is_person
        z = keep_speaker_sys_clusters and keep_person_sys_clusters and is_speaker and is_person

        if w or x or y or z:
            sys_clusters.append(sys_cluster)
    
    if heuristic_speaker_resolution:
        print("\theuristic speaker clustering")
        if heuristic_pronoun_resolution:
            print("\theuristic pronoun resolution")
        heuristic_clusters = heuristic_coreference_resolution(document, mention_tags, spacy_document, min_speaker_sim=min_speaker_sim, max_speaker_merges=max_speaker_merges, pronoun_resolution=heuristic_pronoun_resolution)
        sys_clusters = merge_clusters(sys_clusters + heuristic_clusters)
    
    print(f"{len(sys_clusters)} sys clusters")

    evaluation = evaluate_coreference(list(gold_clusters.values()), sys_clusters)
    return {"evaluation": evaluation, "gold_clusters": gold_clusters, "sys_clusters": sys_clusters, "coref_dataframe": coref_df, "document": document, "coref_result": coref_result, "mention_tags": mention_tags, "start_line_index": start_line_index, "end_line_index": end_line_index}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate coreference resolution by joining parsed elements", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--parsed_script", help="parsed text file", dest="parsed", type=str, default="data/annotation/basterds.script_parsed.txt")
    parser.add_argument("--matched_gold_coref", help="csv file containing gold coreference annotations that have been matched with parsed script", dest="coref", type=str, default="data/annotation/basterds.coref.mapped.csv")
    parser.add_argument("--device", help="cuda device index, -1 for cpu", dest="device", default=-1, type=int)
    parser.add_argument("--use_speaker_sep", help="set to use 'says' between character names and utterance", dest="speaker", action="store_true")
    parser.add_argument("--keep_only_speaker_gold_clusters", help="set to use only those gold clusters that refer to speaking characters", dest="keep_speaker", action="store_true")
    parser.add_argument("--remove_singleton_gold_clusters", help="set to remove gold clusters that have only one mention", dest="remove_singleton", action="store_true")
    args = parser.parse_args()

    parsed_file = args.parsed
    coref_file = args.coref
    cuda_device = args.device
    use_speaker_sep = args.speaker
    keep_only_speaker_gold_clusters = args.keep_speaker
    remove_singleton_gold_clusters = args.remove_singleton

    evaluate_coreference_by_joining_elements(parsed_file, coref_file, cuda_device, use_speaker_sep, keep_only_speaker_gold_clusters, remove_singleton_gold_clusters)