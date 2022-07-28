import re
import spacy
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

def nec_f1_score(gold: set, sys: set):
    u, v, w = len(gold.intersection(sys)), len(gold), len(sys)
    if v:
        return 2 * u / (v + w)
    else:
        return int(w == 0)

def evaluate_coreference_nec(gold_entity_to_mentions, sys_clusters, coref_df, document):
    pronouns = "I, me, my, mine, myself, We, us, our, ours, ourselves, you, your, yours, yourself, yourselves, he, him, his, himself, she, her, hers, herself, it, its, itself, they, them, their, theirs, themself, themselves".lower().split(", ")

    coref_df.PRONOUN |= coref_df.mention.str.lower().isin(pronouns)
    document_pronouns = set(coref_df[coref_df.PRONOUN].mention.str.lower().unique()).union(pronouns)

    spacy_nlp = spacy.load("en_core_web_sm")
    spacy_document = spacy_nlp(document)

    _gold_entity_to_mentions = defaultdict(set)
    gold_entity_to_name_mentions = defaultdict(set)
    gold_entity_to_names = defaultdict(set)
    gold_entity_to_pronoun_mentions = defaultdict(set)
    gold_entity_to_nominal_mentions = defaultdict(set)
    sys_pronoun_clusters = []
    sys_nominal_clusters = []
    sys_name_clusters = []

    print("finding gold names, pronouns and nominals mentions")
    for entity, df in coref_df.groupby("entityLabel"):
        name_mentions = set()
        pronoun_mentions = set()
        nominal_mentions = set()

        for _, row in df.iterrows():
            mention = (row.mention_start, row.mention_end)
            if row.PRONOUN:
                pronoun_mentions.add(mention)
            elif row.NOMINAL:
                nominal_mentions.add(mention)
            else:
                name_mentions.add(mention)
            
        name_mentions.intersection_update(gold_entity_to_mentions[entity])
        pronoun_mentions.intersection_update(gold_entity_to_mentions[entity])
        nominal_mentions.intersection_update(gold_entity_to_mentions[entity])

        if name_mentions:
            gold_entity_to_name_mentions[entity] = name_mentions
            gold_entity_to_pronoun_mentions[entity] = pronoun_mentions
            gold_entity_to_nominal_mentions[entity] = nominal_mentions
            _gold_entity_to_mentions[entity] = name_mentions.union(pronoun_mentions).union(nominal_mentions)

    gold_entity_to_mentions = _gold_entity_to_mentions
    entities = list(gold_entity_to_mentions.keys())

    print("finding names of entities")
    for entity in entities:
        name_mentions = gold_entity_to_name_mentions[entity]
        names = set()

        for i, j in name_mentions:
            char_begin = spacy_document[i].idx
            char_end = spacy_document[j].idx + len(spacy_document[j])
            text = document[char_begin: char_end]
            text = re.sub("\s+", " ", text).strip()
            spacy_text = spacy_nlp(text)
            head_token = [token for token in spacy_text if token.head == token][0]

            for noun_chunk in spacy_text.noun_chunks:
                contains_proper_noun = any([token.pos_ == "PROPN" for token in noun_chunk])
                if contains_proper_noun and not noun_chunk.text.islower() and head_token in noun_chunk:
                    names.add(noun_chunk.text.lower())
            
            names.add(text.lower())
        
        gold_entity_to_names[entity] = names

    print("finding sys names, pronouns and nominals mentions")
    for mentions in sys_clusters:
        pronoun_mentions = set()
        nominal_mentions = set()
        name_mentions = set()

        for i, j in mentions:
            char_begin = spacy_document[i].idx
            char_end = spacy_document[j].idx + len(spacy_document[j])
            text = document[char_begin: char_end]
            text = re.sub("\s+", " ", text).strip()
            spacy_text = spacy_nlp(text)
            head_token = [token for token in spacy_text if token.head == token][0]

            if text.lower() in document_pronouns:
                pronoun_mentions.add((i, j))
            elif head_token.pos_ == "PROPN":
                name_mentions.add((i, j))
            else:
                nominal_mentions.add((i, j))

        sys_pronoun_clusters.append(pronoun_mentions)
        sys_nominal_clusters.append(nominal_mentions)
        sys_name_clusters.append(name_mentions)

    nec_f1_mat = np.zeros((len(entities), len(sys_clusters)))

    print("calculating nec f1")
    for i, entity in enumerate(entities):
        for j, sys_mentions in enumerate(sys_clusters):
            for k, l in sys_mentions:
                char_begin = spacy_document[k].idx
                char_end = spacy_document[l].idx + len(spacy_document[l])
                text = document[char_begin: char_end]
                text = re.sub("\s+", " ", text).strip().lower()
                contains_gold_name = any([re.search("(^|\s)" + re.escape(name) + "(\s|$)", text) is not None for name in gold_entity_to_names[entity]])

                if contains_gold_name:
                    nec_f1_mat[i, j] = nec_f1_score(gold_entity_to_mentions[entity], sys_mentions)
                    break

    row_ind, col_ind = linear_sum_assignment(nec_f1_mat, maximize=True)
    nec_f1 = nec_f1_mat[row_ind, col_ind].sum()/len(entities)
    per_chains_missed = 100*(nec_f1_mat[row_ind, col_ind] == 0).sum()/len(entities)

    nec_pronoun_f1 = 0
    nec_nominal_f1 = 0
    nec_name_f1 = 0

    for r, c in zip(row_ind, col_ind):
        entity = entities[r]
        nec_pronoun_f1 += nec_f1_score(gold_entity_to_pronoun_mentions[entity], sys_pronoun_clusters[c])
        nec_nominal_f1 += nec_f1_score(gold_entity_to_nominal_mentions[entity], sys_nominal_clusters[c])
        nec_name_f1 += nec_f1_score(gold_entity_to_name_mentions[entity], sys_name_clusters[c])

    nec_pronoun_f1 /= len(entities)
    nec_nominal_f1 /= len(entities)
    nec_name_f1 /= len(entities)

    print(f"NEC F1 = {nec_f1:.4f}, chains missed = {per_chains_missed:.2f}%")
    print(f"NEC F1 for pronouns = {nec_pronoun_f1:.4f}, nominals = {nec_nominal_f1:.4f}, names = {nec_name_f1:.4f}")

    nec_result = {"nec_F1": nec_f1, "nec_per_chains_missed": per_chains_missed, "nec_pronoun_F1": nec_pronoun_f1, "nec_nominal_F1": nec_nominal_f1, "nec_name_F1": nec_name_f1}
    meta_info = {"gold_entities": entities, "gold_ind": row_ind.tolist(), "sys_ind": col_ind.tolist()}
    nec_result["meta_info"] = meta_info
    return nec_result