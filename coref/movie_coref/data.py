"""Data structure for movie coreference for fine-tuning worl-level coreference
models
"""

import collections
import jsonlines
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers.tokenization_utils import PreTrainedTokenizer

class Mention:
    """Mention objects represent mentions with head information. It contains
    three integers: start token index, end token index, and head token index
    of the mention. The end token index is inclusive. start <= head <= end.
    """
    def __init__(self, begin: int, end: int, head: int) -> None:
        self.begin = begin
        self.end = end
        self.head = head

    def __hash__(self) -> int:
        return hash((self.begin, self.end))

    def __lt__(self, other: "Mention") -> bool:
        return (self.begin, self.end) < (other.begin, other.end)

    def __repr__(self) -> str:
        return f"({self.begin},{self.end})"

class CorefDocument:
    """CorefDocument objects represent coreference-annotated and parsed movie
    script. It contains the following attributes: movie, rater, tokens, parse
    tags, part-of-speech tags, named entity tags, speaker tags, sentence
    offsets, and clusters. Clusters is a dictionary of character names to set
    of Mention objects.
    """
    def __init__(self, json: dict[str, any]) -> None:
        self.movie: str = json["movie"]
        self.rater: str = json["rater"]
        self.token: list[str] = json["token"]
        self.parse: list[str] = json["parse"]
        self.pos: list[str] = json["pos"]
        self.ner: list[str] = json["ner"]
        self.speaker: list[str] = json["speaker"]
        self.sentence_offsets: list[list[int]] = json["sent_offset"]
        self.clusters: dict[str, set[Mention]] = {}
        for character, mentions in json["clusters"].items():
            mentions = set([Mention(*x) for x in mentions])
            self.clusters[character] = mentions
    
    def __repr__(self) -> str:
        desc = "Script\n=====\n\n"
        for i, j in self.sentence_offsets:
            sentence = self.token[i: j]
            desc += f"{sentence}\n"
        desc += "\n\nClusters\n========\n\n"
        for character, mentions in self.clusters.items():
            desc += f"{character}\n"
            sorted_mentions = sorted(mentions)
            mention_texts = []
            for mention in sorted_mentions:
                mention_text = " ".join(
                    self.token[mention.begin: mention.end + 1])
                mention_head = self.token[mention.head]
                mention_texts.append(f"{mention_text} ({mention_head})")
            n_rows = math.ceil(len(mention_texts)/3)
            for i in range(n_rows):
                row_mention_texts = mention_texts[i * 3: (i + 1) * 3]
                row_desc = "     ".join(
                    [f"{mention_text:25s}" for mention_text in row_mention_texts])
                desc += row_desc + "\n"
            desc += "\n"
        return desc

class CorefCorpus:
    """CorefCorpus is a list of CorefDocuments.
    """
    def __init__(self, file: str | None = None) -> None:
        self.documents: list[CorefDocument] = []
        if file is not None:
            with jsonlines.open(file) as reader:
                for json in reader:
                    self.documents.append(CorefDocument(json))
    
    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, i) -> CorefDocument:
        return self.documents[i]

class CharacterRecognitionDataset(Dataset):
    """PyTorch dataset for the character recognition model. It contains the
    following attributes: `subtoken_ids` tensor, `attention_mask` tensor,
    `token_offset` tensor, `parse_tag_ids` tensor, `label_ids` tensor, and
    `parse_tag_to_id` dictionary.
    """
    def __init__(self, corpus: CorefCorpus, tokenizer: PreTrainedTokenizer,
                 seq_length: int, obey_scene_boundaries: bool) -> None:
        """Initializer for the character recognition model data.

        Args:
            corpus: CorefCorpus
            tokenizer: Transformer tokenizer
            seq_length: maximum number of tokens (not sub-tokens) in a sequence
            obey_scene_boundaries: if true, sequences do not cross scene
            boundaries
        """
        super().__init__()
        tokens_list: list[list[str]] = []
        labels_list: list[torch.Tensor] = []
        parse_tags_list: list[torch.Tensor] = []
        self.parse_tag_to_id = collections.defaultdict(int)
        for i, parse_tag in enumerate("SNCDE"):
            self.parse_tag_to_id[parse_tag] = i + 1

        for document in corpus:
            tokens = document.token
            labels = torch.zeros(len(tokens), dtype=int)
            parse_tags = document.parse
            for mentions in document.clusters.values():
                for mention in mentions:
                    labels[mention.head] = 1

            # Find token-level scene boundaries
            if obey_scene_boundaries:
                scene_boundaries = np.zeros(len(tokens), dtype=int)
                found_content_tag = False
                i = 0
                while i < len(tokens):
                    if parse_tags[i] == "S":
                        if found_content_tag:
                            scene_boundaries[i] = 1
                            found_content_tag = False
                        while i < len(tokens) and parse_tags[i] == "S":
                            i += 1
                    else:
                        if parse_tags[i] in "ND":
                            found_content_tag = True
                        i += 1

            # Create movie parse tensor
            parse_tag_tensor = torch.zeros(len(tokens), dtype=int)
            for i, tag in enumerate(parse_tags):
                parse_tag_tensor[i] = self.parse_tag_to_id[tag]

            # Segment document into sequences
            i = 0
            while i < len(tokens):
                end = i + seq_length
                if obey_scene_boundaries and (
                   np.any(scene_boundaries[i: end] == 1)):
                    end = (i + np.nonzero(scene_boundaries[i: end] == 1)[0][0]
                           + 1)
                tokens_list.append(tokens[i: end])
                labels_list.append(labels[i: end])
                parse_tags_list.append(parse_tag_tensor[i : end])
                i = end

        # Find token character offsets
        token_char_offset_list: list[list[tuple[int, int]]] = []
        text_list: list[str] = []
        max_n_tokens_per_sequence = -np.inf
        for tokens in tokens_list:
            token_char_offset: list[tuple[int, int]] = []
            c = 0
            for token in tokens:
                token_char_offset.append((c, c + len(token)))
                c += len(token) + 1
            token_char_offset_list.append(token_char_offset)
            text_list.append(" ".join(tokens))
            max_n_tokens_per_sequence = max(max_n_tokens_per_sequence,
                                            len(tokens))

        # Encode
        encoding = tokenizer(text_list, padding="longest", return_tensors="pt",
                             return_offsets_mapping=True,
                             return_attention_mask=True)
        
        # Find token to subtoken offset
        token_offset = torch.zeros(
            (len(text_list), max_n_tokens_per_sequence, 2), dtype=int)
        for i, (subtoken_char_offset, token_char_offset, attention_mask) in (
            enumerate(zip(encoding["offset_mapping"], token_char_offset_list,
                          encoding["attention_mask"]))):
            j, k = 0, 0
            n_subtokens = attention_mask.sum()
            while j < n_subtokens and k < len(token_char_offset):
                if subtoken_char_offset[j, 0] == subtoken_char_offset[j, 1]:
                    j += 1
                else:
                    assert (subtoken_char_offset[j, 0] == 
                            token_char_offset[k][0]), (
                                "Begin char offset of token and subtoken "
                                "don't match")
                    l = j
                    while l < n_subtokens and (
                        subtoken_char_offset[l, 1] != token_char_offset[k][1]):
                        l += 1
                    assert (subtoken_char_offset[l, 1] == 
                            token_char_offset[k][1]), (
                                "End char offset of token and subtoken "
                                "don't match")
                    token_offset[i, k, 0] = j
                    token_offset[i, k, 1] = l
                    k += 1
                    j = l + 1

        self.subtoken_ids = encoding["input_ids"]
        self.attention_mask = encoding["attention_mask"]
        self.token_offset = token_offset
        self.parse_tag_ids = pad_sequence(parse_tags_list, batch_first=True,
                                          padding_value=0)
        self.label_ids = pad_sequence(labels_list, batch_first=True,
                                      padding_value=0)
    
    def __len__(self) -> int:
        return len(self.subtoken_ids)
    
    def __getitem__(self, i: int) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
              torch.Tensor]):
        return (self.subtoken_ids[i], self.attention_mask[i],
                self.token_offset[i], self.parse_tag_ids[i], self.label_ids[i])