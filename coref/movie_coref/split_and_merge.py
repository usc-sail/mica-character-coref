"""Functions to split documents and merge coreference predictions from overlapping documents.
"""

import numpy as np

def split_screenplay(self, document: dict[str, any], split_len: int, overlap_len: int):
        """Split screenplay document into smaller documents.

        Args:
            document: Original screenplay document.
            split_len: Length of the smaller documents in words.
            overlap_len: Number of overlapping words.
        
        Returns:
            Generator of documents.
        """
        doc_offsets: list[tuple[int, int]] = []
        segment_boundaries = np.zeros(len(document["cased_words"]), dtype=int)
        i = 0
        parse_tags = document["parse"]
        sentence_offsets = np.array(document.sentence_offsets)
        while i < len(document["cased_words"]):
            if parse_tags[i] in "SNC":
                j = i + 1
                while j < len(document["cased_words"]) and (
                    parse_tags[j] == parse_tags[i]):
                    j += 1
                segment_boundaries[i] = 1
                i = j
            else:
                i += 1
        segment_boundaries[0] = 1
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
        for k, (i, j) in enumerate(doc_offsets):
            _document = CorefDocument()
            _document.movie = document.movie + f"_{k + 1}"
            _document.rater = document.rater
            _document["cased_words"] = document["cased_words"][i: j]
            _document.parse = document.parse[i: j]
            _document.parse_ids = [parse_labelset[x] for x in _document.parse]
            _document.pos = document.pos[i: j]
            _document.pos_ids = [pos_labelset[x] for x in _document.pos]
            _document.ner = document.ner[i: j]
            _document.ner_ids = [ner_labelset[x] for x in _document.ner]
            _document.is_pronoun = document.is_pronoun[i: j]
            _document.is_punctuation = document.is_punctuation[i: j]
            _document.speaker = document.speaker[i: j]
            si = np.nonzero(sentence_offsets[:,0] == i)[0][0]
            sj = np.nonzero(sentence_offsets[:,1] == j - 1)[0][0] + 1
            _document.sentence_offsets = (sentence_offsets[si: sj] - sentence_offsets[si, 0]
                ).tolist()
            clusters: dict[str, set[Mention]] = collections.defaultdict(set)
            n_mentions = 0
            for character, mentions in document.clusters.items():
                for mention in mentions:
                    assert (mention.end < i or i <= mention.begin <= mention.end < j or 
                        j <= mention.begin), "Mention crosses document boundaries"
                    if i <= mention.begin <= mention.end < j:
                        mention.begin -= i
                        mention.end -= i
                        mention.head -= i
                        clusters[character].add(mention)
                        n_mentions += 1
            _document.clusters = clusters
            _document.word_cluster_ids = document.word_cluster_ids[i: j]
            _document.word_head_ids = document.word_head_ids[i: j]
            self._log(f"{_document.movie}: {len(_document["cased_words"])} words, "
                f"{n_mentions} mentions, {len(_document.clusters)} clusters")
            yield _document