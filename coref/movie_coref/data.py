"""Data structure for movie coreference for fine-tuning worl-level coreference
models
"""

import jsonlines
import math
import numpy as np
import string
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
        pronouns = "you i he my him me his yourself mine your her she".split()
        punctuation = list(string.punctuation)
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
        self.is_pronoun: list[bool] = [t.lower() in pronouns for t in self.token]
        self.is_punctuation: list[bool] = [t in punctuation for t in self.token]
        self._data: dict[str, any] = {}
    
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

class LabelSet:

    def __init__(self, labels: list[str], add_other: bool = True) -> None:
        """Initializer for label sets.

        Args:
            labels: list of class label names.
            add_other: If true, add a other class.
        """
        self._add_other = add_other
        self._labels = labels
        self._other_label = "<O>"
        assert self._other_label not in self._labels, (
            f"{self._other_label} cannot be a label")
        if add_other:
            self._labels.append(self._other_label)
        self._label_to_id: dict[str, int] = {}
        self._id_to_label: dict[int, str] = {}
        for i, label in enumerate(self._labels):
            self._label_to_id[label] = i
            self._id_to_label[i] = label
    
    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, key: int | str) -> int | str:
        """Get the label (label_id) from label_id (label)"""
        if isinstance(key, str):
            key = self.convert_label_to_key(key)
            return self._label_to_id[key]
        elif isinstance(key, int):
            return self._id_to_label[key]
        else:
            raise TypeError
    
    def convert_label_to_key(self, label: str) -> str:
        """Convert label to label key"""
        if label not in self._labels:
            if self._add_other:
                return self.other_label
            else:
                raise KeyError(f"label={label} not found in label set")
        else:
            return label
    
    @property
    def other_id(self) -> int:
        """Return label_id of other_label"""
        return self.__getitem__(self._other_label)
    
    @property
    def other_label(self) -> str:
        """Return other_label"""
        return self._other_label

    def __repr__(self) -> str:
        desc_list = []
        for i, l in self._id_to_label.items():
            desc_list.append(f"{i}:{l}")
        return " ".join(desc_list)

class PosLabelSet(LabelSet):
    def convert_label_to_key(self, label: str) -> str:
        if label.startswith("NN"):
            return "NOUN"
        elif label.startswith("VB"):
            return "VERB"
        elif label.startswith("JJ"):
            return "ADJECTIVE"
        elif label.startswith("RB"):
            return "ADVERB"
        else:
            return self.other_label

class CharacterRecognitionDataset(Dataset):
    """PyTorch dataset for the character recognition model.
    """
    def __init__(self, corpus: CorefCorpus, tokenizer: PreTrainedTokenizer,
                 seq_length: int, obey_scene_boundaries: bool,
                 label_type = "head") -> None:
        """Initializer for the character recognition model data.

        Args:
            corpus: CorefCorpus
            tokenizer: Transformer tokenizer
            seq_length: maximum number of tokens (not sub-tokens) in a sequence
            obey_scene_boundaries: if true, sequences do not cross scene
            boundaries
            label_type: "head" or "span"
        """
        super().__init__()
        assert label_type in ["head", "span"], (
            f"label_type should be 'head' or 'span'. Given '{label_type}'")
        
        # Label sets
        if label_type == "head":
            self.label_set = LabelSet(["HEAD"])
        else:
            self.label_set = LabelSet(["B-SPAN", "I-SPAN"])
        self.parse_tag_set = LabelSet(list("SNCDE"))
        self.pos_tag_set = PosLabelSet(["NOUN", "VERB", "ADJECTIVE", "ADVERB"])
        self.ner_tag_set = LabelSet(["PERSON", "ORG", "GPE", "LOC"])

        # Tensors list
        tokens_list: list[list[str]] = []
        labels_list: list[torch.Tensor] = []
        parse_tags_list: list[torch.Tensor] = []
        pos_tags_list: list[torch.Tensor] = []
        ner_tags_list: list[torch.Tensor] = []
        is_pronoun_list: list[torch.Tensor] = []
        is_punctuation_list: list[torch.Tensor] = []

        for document in corpus:
            tokens = document.token
            parse_tags = document.parse
            pos_tags = document.pos
            ner_tags = document.ner
            is_pronoun = document.is_pronoun
            is_punctuation = document.is_punctuation

            # Create labels tensor
            labels = torch.full(
                (len(tokens),), self.label_set.other_id, dtype=int)
            for mentions in document.clusters.values():
                for mention in mentions:
                    if label_type == "head":
                        labels[mention.head] = self.label_set["HEAD"]
                    else:
                        labels[mention.begin] = self.label_set["B-SPAN"]
                        labels[mention.begin + 1: mention.end + 1] = (
                            self.label_set["I-SPAN"])

            # Create movie parse tensor
            parse_tag_tensor = torch.zeros(len(tokens), dtype=int)
            for i, tag in enumerate(parse_tags):
                parse_tag_tensor[i] = self.parse_tag_set[tag]
            
            # Create movie pos tensor
            pos_tag_tensor = torch.zeros(len(tokens), dtype=int)
            for i, tag in enumerate(pos_tags):
                pos_tag_tensor[i] = self.pos_tag_set[tag]

            # Create movie ner tensor
            ner_tag_tensor = torch.zeros(len(tokens), dtype=int)
            for i, tag in enumerate(ner_tags):
                ner_tag_tensor[i] = self.ner_tag_set[tag]

            # Create movie is pronoun tensor
            is_pronoun_tensor = torch.FloatTensor(is_pronoun)

            # Create movie is punctuation tensor
            is_punctuation_tensor = torch.FloatTensor(is_punctuation)

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

            # Segment document into sequences
            i = 0
            while i < len(tokens):
                end = i + seq_length
                if obey_scene_boundaries and (
                   np.any(scene_boundaries[i + 1: end] == 1)):
                    end = i + np.nonzero(
                        scene_boundaries[i + 1: end] == 1)[0][0] + 1
                tokens_list.append(tokens[i: end])
                labels_list.append(labels[i: end])
                parse_tags_list.append(parse_tag_tensor[i: end])
                pos_tags_list.append(pos_tag_tensor[i: end])
                ner_tags_list.append(ner_tag_tensor[i: end])
                is_pronoun_list.append(is_pronoun_tensor[i: end])
                is_punctuation_list.append(is_punctuation_tensor[i: end])
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
        self.label_ids = pad_sequence(labels_list, batch_first=True,
                                      padding_value=self.label_set.other_id)
        self.parse_tag_ids = pad_sequence(
            parse_tags_list, batch_first=True,
            padding_value=self.parse_tag_set.other_id)
        self.pos_tag_ids = pad_sequence(
            pos_tags_list, batch_first=True,
            padding_value=self.pos_tag_set.other_id)
        self.ner_tag_ids = pad_sequence(
            ner_tags_list, batch_first=True,
            padding_value=self.ner_tag_set.other_id)
        self.is_pronoun = pad_sequence(
            is_pronoun_list, batch_first=True, padding_value=0)
        self.is_punctuation = pad_sequence(
            is_punctuation_list, batch_first=True, padding_value=0)
    
    def __len__(self) -> int:
        return len(self.subtoken_ids)
    
    def __getitem__(self, i: int) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
              torch.Tensor]):
        return dict(subtoken_ids=self.subtoken_ids[i],
                    attention_mask=self.attention_mask[i],
                    token_offset=self.token_offset[i],
                    labels=self.label_ids[i],
                    parse_ids=self.parse_tag_ids[i],
                    pos_ids=self.pos_tag_ids[i],
                    ner_ids=self.ner_tag_ids[i],
                    is_pronoun=self.is_pronoun[i],
                    is_punctuation=self.is_punctuation[i])