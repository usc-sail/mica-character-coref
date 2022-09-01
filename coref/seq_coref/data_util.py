"""Functions to perform data transformations, loading, and saving
"""

from mica_text_coref.coref.seq_coref import data
from mica_text_coref.coref.seq_coref import mention_tree

import tqdm
from typing import Callable

def load_data(jsonlines_file: str, keep_singletons: bool = False) -> \
    data.CorefCorpus:
    """Load CorefCorpus from jsonlines file and remove overlapping mentions
    from coreference clusters. If the cluster becomes a singleton after
    removing the overlapping mentions, remove or keep it according to the
    keep_singletons argument.
    """
    corpus = data.CorefCorpus(jsonlines_file)
    for document in corpus.documents:
        clusters = []
        for cluster in document.clusters:
            forest = mention_tree.create_mention_forest(cluster)
            mentions: set[data.Mention] = set()
            for mention_node in forest:
                for node in (mention_tree
                .find_largest_subset_of_non_overlapping_mention_nodes(
                    mention_node)):
                    mentions.add(node.mention)
            if keep_singletons or len(mentions) > 1:
                clusters.append(mentions)
        document.clusters = clusters
    return corpus

def remap_spans(corpus: data.CorefCorpus, 
                tokenize_fn: Callable[[str], list[str]]) -> data.CorefCorpus:
    """Apply the tokenize_fn to each word of the corpus's documents and adjust
    the indices of the mentions in the coreference clusters. The mentions of
    the named entity and constituent dictionaries are also adjusted.
    """
    new_corpus = data.CorefCorpus()
    for document in tqdm.tqdm(corpus.documents):
        new_document = data.CorefDocument()
        mapping: list[int] = [0]
        n_tokens = 0
        
        for sentence, speakers in zip(document.sentences, document.speakers):
            new_sentence = []
            new_speakers = []
            for word, spk in zip(sentence, speakers):
                if word == "-LRB-":
                    word = "("
                elif word == "-RRB-":
                    word = ")"
                tokens = tokenize_fn(word)
                new_sentence.extend(tokens)
                new_speakers.extend([spk for _ in range(len(tokens))])
                n_tokens += len(tokens)
                mapping.append(n_tokens)
            new_document.sentences.append(new_sentence)
            new_document.speakers.append(new_speakers)

        for cluster in document.clusters:
            new_cluster: set[data.Mention] = set()
            for mention in cluster:
                new_begin = mapping[mention.begin]
                new_end = mapping[mention.end + 1] - 1
                new_mention = data.Mention(new_begin, new_end)
                new_cluster.add(new_mention)
            new_document.clusters.append(new_cluster)

        for mention, ner_tag in document.named_entities.items():
            new_begin = mapping[mention.begin]
            new_end = mapping[mention.end + 1] - 1
            new_mention = data.Mention(new_begin, new_end)
            new_document.named_entities[new_mention] = ner_tag
        
        for mention, constituency_tag in document.constituents.items():
            new_begin = mapping[mention.begin]
            new_end = mapping[mention.end + 1] - 1
            new_mention = data.Mention(new_begin, new_end)
            new_document.constituents[new_mention] = constituency_tag
        
        new_document.doc_key = document.doc_key
        new_corpus.documents.append(new_document)

    return new_corpus