"""Functions to select a representative mention per coreference chain."""

from mica_text_coref.coref.seq_coref import data

import numpy as np


def representative_mention(
    cluster: set[data.Mention], document: data.CorefDocument) -> data.Mention:
    """Choose a representative mention from the cluster.
    """
    sorted_cluster = sorted(cluster)
    is_person = np.full(len(sorted_cluster), dtype=bool, fill_value=False)
    is_noun_phrase = np.full(len(sorted_cluster), dtype=bool, fill_value=False)
    indices = np.arange(len(sorted_cluster), dtype=int)

    for i, mention in enumerate(sorted_cluster):
        is_person[i] = mention in document.named_entities and (
            document.named_entities[mention] == "PERSON")
        is_noun_phrase[i] = mention in document.constituents and (
            document.constituents[mention] == "NP")
    
    if is_person.any():
        first_person_mention = sorted_cluster[indices[is_person][0]]
        return first_person_mention
    elif is_noun_phrase.any():
        first_noun_phrase_mention = sorted_cluster[indices[is_noun_phrase][0]]
        return first_noun_phrase_mention
    else:
        return sorted_cluster[0]