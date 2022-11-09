"""Functions to split word-level json documents and merge coreference predictions from overlapping 
documents.
"""

import collections
import numpy as np
import re

def split_screenplay(document: dict[str, any], split_len: int, overlap_len: int):
    """Split screenplay document into smaller documents.

    Args:
        document: Original screenplay document.
        split_len: Length of the smaller documents in words.
        overlap_len: Number of overlapping words.

    Returns:
        Generator of documents
    """
    doc_offsets: list[tuple[int, int]] = []

    # Find segment boundaries. A segment boundary is the start of the script or of a scene 
    # header, description, or speaker.
    i = 0
    parse_tags = document["parse"]
    segment_boundaries = np.zeros(len(document["cased_words"]), dtype=int)
    while i < len(document["cased_words"]):
        if parse_tags[i] in "SNC":
            j = i + 1
            while j < len(document["cased_words"]) and (parse_tags[j] == parse_tags[i]):
                j += 1
            segment_boundaries[i] = 1
            i = j
        else:
            i += 1
    segment_boundaries[0] = 1

    # Find offsets of the smaller documents. A smaller document should start at a segment boundary.
    i = 0
    while i < len(document["cased_words"]):
        j = min(i + split_len, len(document["cased_words"]))
        if j < len(document["cased_words"]):
            while j >= i and segment_boundaries[j] == 0:
                j -= 1
            k = i + split_len - overlap_len
            while k >= i and segment_boundaries[k] == 0:
                k -= 1
            nexti = k
        else:
            nexti = j
        assert i < nexti, "Document length is 0!"
        doc_offsets.append((i, j))
        i = nexti

    # Numify original sentence offsets
    sentence_offsets = np.array(document["sent_offset"])
    
    # Create the smaller documents
    for k, (i, j) in enumerate(doc_offsets):
        # Populate general fields
        _document: dict[str, any] = {}
        _document["movie"] = document["movie"] + f"_{k + 1}"
        _document["rater"] = document["rater"]
        _document["token"] = document["token"][i: j]
        _document["pos"] = document["pos"][i: j]
        _document["ner"] = document["ner"][i: j]
        _document["parse"] = document["parse"][i: j]
        _document["speaker"] = document["speaker"][i: j]

        # Populate word-level coref model specific fields
        _document["document_id"] = document["document_id"] + f"_{k + 1}"
        _document["cased_words"] = document["cased_words"][i: j]
        _document["offset"] = [i, j]

        # Populate clusters
        clusters: dict[str, set[tuple[int, int, int]]] = collections.defaultdict(set)
        n_mentions = 0
        for character, mentions in document["clusters"].items():
            for begin, end, head in mentions:
                assert end < i or i <= begin <= end < j or j <= begin, (
                    "Mention crosses segment boundaries")
                if i <= begin <= end < j:
                    begin -= i
                    end -= i
                    head -= i
                    clusters[character].add((begin, end, head))
                    n_mentions += 1
        _clusters: dict[str, list[list[int]]] = {}
        for character, mentions in clusters.items():
            _mentions = [[begin, end, head] for begin, end, head in mentions]
            _clusters[character] = _mentions
        _document["clusters"] = _clusters

        # Populate sentence offset
        si = np.nonzero(sentence_offsets[:,0] == i)[0][0]
        sj = np.nonzero(sentence_offsets[:,1] == j - 1)[0][0] + 1
        _document["sent_offset"] = (sentence_offsets[si: sj] - sentence_offsets[si, 0]).tolist()

        # Populate sentence ids
        parse_ids = []
        i, c = 0, 0
        while i < len(_document["parse"]):
            j = i + 1
            while j < len(_document["parse"]) and _document["parse"][j] == _document["parse"][i]:
                j += 1
            parse_ids.extend([c] * (j - i))
            c += 1
            i = j
        _document["sent_id"] = parse_ids

        print(f"{_document['document_id']}: {len(_document['cased_words'])} words, "
            f"{n_mentions} mentions, {len(_document['clusters'])} clusters")
        yield _document

# def combine_screenplays(documents: list[dict[str, any]], starts: list[int],
#     ends: list[int]) -> dict[str, any]:
#     """Combine predictions from smaller screenplays.

#     Args:
#         documents: List of smaller documents.
#         starts: List of integers specifying the start of the document.
#         ends: List of integers specifying the end of the document.

#     Returns:
#         Combined document.
#     """
#     assert len(documents) > 0, "No documents provided"
#     assert len(documents) == len(starts) == len(ends), "Number of documents and length of starts and ends list should equal"
#     _document: dict[str, any] = {}
#     _document["document_id"] = re.sub("_\d+$", "", documents[0]["document_id"])
#     cased_words, parse_tags, speakers = [], [], []
#     gold_clusters = collections.defaultdict(set)
#     for document, i, j in zip(documents, starts, ends):
#         k = len(cased_words) - i
#         cased_words.extend(document["cased_words"][k: j])
#         parse_tags.extend(document["parse"][k: j])
#         speakers.extend(document["speakers"][k: j])
#         for character, mentions in document["clusters"].items():
#             for begin, end, head in mentions:
#                 gold_clusters[character].add((begin + i, end + i, head + i))