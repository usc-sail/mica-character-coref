"""Functions to perform data transformations, loading, and saving
"""

from mica_text_coref.coref.seq_coref import data
from mica_text_coref.coref.seq_coref import mention_tree

import copy
import numpy as np
import tqdm
from typing import Callable

def remove_overlaps(corpus: data.CorefCorpus, 
                    keep_singletons: bool = False) -> data.CorefCorpus:
    """Remove overlapping mentions from coreference clusters. If the cluster
    becomes a singleton after removing the overlapping mentions, remove or keep
    it according to the keep_singletons argument.
    """
    new_corpus = data.CorefCorpus()

    for document in tqdm.tqdm(corpus.documents):
        new_document = copy.deepcopy(document)
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
        
        new_document.clusters = clusters
        new_corpus.documents.append(new_document)
    
    return new_corpus

def remap_spans_word_level(corpus: data.CorefCorpus, 
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

def remap_spans_document_level(
    corpus: data.CorefCorpus, tokenize_fn: Callable[[str], list[str]]) -> (
        data.CorefCorpus):
    """Apply tokenize function at the document level by concatenating all
    words in the documents together. Then adjust the indices of the mentions
    in the coreference clusters, and the named entity and constituencies
    dictionaries. Prefer this over remap_spans_at_word_level if you are using
    transformer-based tokenizers.
    """
    new_corpus = data.CorefCorpus()

    for document in tqdm.tqdm(corpus.documents):
        new_document = data.CorefDocument()
        words = [word for sentence in document.sentences for word in sentence]
        text = " ".join(words)
        tokens = tokenize_fn(text)
        word_characters = "".join(words)
        token_characters = "".join(tokens)
        word_begin_to_word_character = np.zeros(len(words), dtype=int)
        word_end_to_word_character = np.zeros(len(words), dtype=int)
        word_character_to_token_character = np.zeros(len(word_characters),
                                                    dtype=int)
        token_character_to_token_index = np.zeros(len(token_characters),
                                                dtype=int)
        
        c = 0
        for i, word in enumerate(words):
            word_begin_to_word_character[i] = c
            word_end_to_word_character[i] = c + len(word) - 1
            c += len(word)
        
        i, j = 0, 0
        while i < len(word_characters) and j < len(token_characters):
            if word_characters[i] == token_characters[j]:
                word_character_to_token_character[i] = j
                i += 1
                j += 1
            else:
                j += 1
        
        c = 0
        for i, token in enumerate(tokens):
            token_character_to_token_index[c: c + len(token)] = i
            c += len(token)

        def map_begin(word_begin: int) -> int:
            return token_character_to_token_index[
                    word_character_to_token_character[
                        word_begin_to_word_character[word_begin]]]

        def map_end(word_end: int) -> int:
            return token_character_to_token_index[
                    word_character_to_token_character[
                        word_end_to_word_character[word_end]]]
        
        for cluster in document.clusters:
            new_cluster: set[data.Mention] = set()
            for mention in cluster:
                new_begin = map_begin(mention.begin)
                new_end = map_end(mention.end)
                new_mention = data.Mention(new_begin, new_end)
                new_cluster.add(new_mention)
            new_document.clusters.append(new_cluster)

        for mention, ner_tag in document.named_entities.items():
            new_begin = map_begin(mention.begin)
            new_end = map_end(mention.end)
            new_mention = data.Mention(new_begin, new_end)
            new_document.named_entities[new_mention] = ner_tag
        
        for mention, constituency_tag in document.constituents.items():
            new_begin = map_begin(mention.begin)
            new_end = map_end(mention.end)
            new_mention = data.Mention(new_begin, new_end)
            new_document.constituents[new_mention] = constituency_tag
        
        new_sentences = []
        new_speakers = []
        i, j = 0, 0
        for sentence, speakers in zip(document.sentences, document.speakers):
            n_words = len(sentence)
            end = map_end(i + n_words - 1)
            new_sentence = tokens[j: end + 1]
            i += n_words
            j = end + 1
            new_sentences.append(new_sentence)
            new_speakers.append([speakers[0] for _ in range(len(new_sentence))])
        new_document.sentences = new_sentences
        new_document.speakers = new_speakers

        new_document.doc_key = document.doc_key
        new_corpus.documents.append(new_document)
    
    return new_corpus