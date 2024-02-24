"""Functions to split word-level json documents and merge coreference predictions from overlapping documents.
"""
import collections
import numpy as np
import torch
from typing import Any

def split_screenplay(document: dict[str, Any], split_len: int, overlap_len: int, verbose = False):
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
            k = j - overlap_len
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
        _document: dict[str, Any] = {}
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

        if verbose:
            print(f"{_document['document_id']}: {len(_document['cased_words'])} words, {n_mentions} mentions, "
                  f"{len(_document['clusters'])} clusters")
        yield _document

def combine_coref_scores(corefs: list[torch.Tensor], inds: list[torch.Tensor], overlap_lens: list[int],
                         strategy: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine corefs and inds into a single coref and ind tensor.
    
    Args:
        corefs: list[Tensor[*, k + 1]]
        inds: list[Tensor[*, k]]
        overlap_lens: list[int]
        strategy: Can be one of "pre", "post", "avg", "max", "min", or "none"
        
    Return:
        coref: Tensor[n, 2k + 1]
        top_indices: Tensor[n, 2k]
    """
    # Assertions
    assert strategy in ["pre", "post", "avg", "max", "min", "none"], "Invalid strategy"
    assert len(corefs) > 0, "Number of coref tensors should be atleast 1"
    assert len(corefs) == len(inds), "Number of coref tensors should equal number of indices tensors"
    if len(corefs) == 1: return corefs[0], inds[0]
    assert len(overlap_lens) == len(corefs) - 1, ("Number of overlap lengths should equal one less than the number of "
                                                  "coref tensors")

    # Intialize
    n = sum([len(coref) - overlap_len for coref, overlap_len in zip(corefs[:-1], overlap_lens)]) + len(corefs[-1])
    k = inds[0].shape[1]
    device = corefs[0].device
    coref = torch.full((n, 2*k), fill_value=-torch.inf, device=device)
    ind = torch.full((n, 2*k), fill_value=-1, device=device)
    coref_start, coref_end = 0, 0
    ext_overlap_lens = overlap_lens + [0, 0]

    # Combine
    for i in range(len(corefs)):
        assert len(corefs[i]) - ext_overlap_lens[i - 1] - ext_overlap_lens[i] > 0, "Atmost two segments should overlap"
        coref_start, coref_end = coref_end, coref_end + len(corefs[i]) - ext_overlap_lens[i - 1] - ext_overlap_lens[i]
        start, end = ext_overlap_lens[i - 1], len(corefs[i]) - ext_overlap_lens[i]

        # Non-overlapping
        coref[coref_start: coref_end, :k] = corefs[i][start: end, 1:]
        ind[coref_start: coref_end, :k] = inds[i][start: end]

        # Overlapping
        coref_start, coref_end = coref_end, coref_end + ext_overlap_lens[i]
        if strategy != "none" and i < len(corefs) - 1:
            for j in range(ext_overlap_lens[i]):
                heads_x, heads_y = inds[i][end + j].tolist(), inds[i + 1][j].tolist()
                scores_x, scores_y = corefs[i][end + j, 1:].tolist(), corefs[i + 1][j, 1:].tolist()
                head_to_score = {h: s for h, s in zip(heads_x, scores_x) if s != -torch.inf}
                for h, s in zip(heads_y, scores_y):
                    if s != -torch.inf:
                        if h in head_to_score:
                            if strategy == "post": head_to_score[h] = s
                            elif strategy == "avg": head_to_score[h] = 0.5 * (head_to_score[h] + s)
                            elif strategy == "max": head_to_score[h] = max(head_to_score[h], s)
                            elif strategy == "min": head_to_score[h] = min(head_to_score[h], s)
                        else: head_to_score[h] = s
                for l, (h, s) in enumerate(head_to_score.items()):
                    coref[coref_start + j, l] = s
                    ind[coref_start + j, l] = h
        else:
            coref[coref_start: coref_end, :k] = corefs[i][end:, 1:]
            ind[coref_start: coref_end, :k] = inds[i][end:]

    # Add dummy
    dummy = torch.zeros((n, 1), device=coref.device)
    coref = torch.cat((dummy, coref), dim=1)
    return coref, ind

def combine_character_scores(character_scores_arr: list[torch.Tensor], overlap_lens: list[int], 
                             strategy: str) -> torch.Tensor:
    """Combine list of character scores arrays into a single character scores array.

    Args:
        character_scores_arr: List of character scores array, each coming from a subdocument.
        overlap_lens: List of number of overlapping words between successive subdocuments
        strategy: Merging strategy, can be one of 'none', 'pre', 'post', 'max', 'min', or 'avg'.
            The behavior of 'none' and 'pre' is the same here.
    
    Returns:
        combined character scores: torch.Tensor
    """
    assert len(character_scores_arr) > 0, "Number of character scores arrays should be greater than 0"
    if len(character_scores_arr) == 1:
        return character_scores_arr[0]
    assert len(overlap_lens) == len(character_scores_arr) - 1, ("Number of overlap lengths should be number of "
                                                                "character scores arrays - 1")
    n = (sum(len(character_scores) - overlap_len
                for character_scores, overlap_len in zip(character_scores_arr[:-1], overlap_lens))
        + len(character_scores_arr[-1]))
    device = character_scores_arr[0].device
    merged_character_scores = torch.zeros(n, device=device, dtype=float) # type: ignore
    ext_overlap_lens = overlap_lens + [0, 0]
    character_start, character_end = 0, 0

    for i in range(len(character_scores_arr)):
        assert len(character_scores_arr[i]) - ext_overlap_lens[i - 1] - ext_overlap_lens[i] > 0, (
            "Atmost two segments should overlap")
        character_start, character_end = character_end, (character_end + len(character_scores_arr[i])
                                                         - ext_overlap_lens[i - 1] - ext_overlap_lens[i])
        start, end = ext_overlap_lens[i - 1], len(character_scores_arr[i]) - ext_overlap_lens[i]
        merged_character_scores[character_start: character_end] = character_scores_arr[i][start: end]
        character_start, character_end = character_end, character_end + ext_overlap_lens[i]
        if i < len(character_scores_arr) - 1 and ext_overlap_lens[i] > 0:
            if strategy == "pre" or strategy == "none":
                merged_character_scores[character_start: character_end] = character_scores_arr[i][-ext_overlap_lens[i]:]
            elif strategy == "post":
                merged_character_scores[character_start: character_end] = character_scores_arr[i + 1][:ext_overlap_lens[i]]
            elif strategy == "avg":
                merged_character_scores[character_start: character_end] = (
                    character_scores_arr[i][-ext_overlap_lens[i]:]
                    + character_scores_arr[i + 1][:ext_overlap_lens[i]])/2
            elif strategy == "max":
                merged_character_scores[character_start: character_end] = torch.max(
                    character_scores_arr[i][-ext_overlap_lens[i]:], character_scores_arr[i + 1][:ext_overlap_lens[i]])
            elif strategy == "min":
                merged_character_scores[character_start: character_end] = torch.min(
                    character_scores_arr[i][-ext_overlap_lens[i]:], character_scores_arr[i + 1][:ext_overlap_lens[i]])

    return merged_character_scores

def combine_head2spans(head2spans: list[dict[int, tuple[int, int, float]]]) -> dict[int, tuple[int, int, float]]:
    """Combine list of head2span dictionaries."""
    head2span = {}
    for h2s in head2spans:
        for head, (start, end, score) in h2s.items():
            if head not in head2span or head2span[head][2] < score:
                head2span[head] = (start, end, score)
    return head2span