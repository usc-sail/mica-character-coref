"""Data structure for movie coreference for fine-tuning worl-level coreference models
"""
import math
import string
import jsonlines
import numpy as np
import torch
from torch.utils.data import DataLoader


pronouns = "you i he my him me his yourself mine your her she".split()
punctuation = list(string.punctuation)


class LabelSet:
    """Class to represent label sets"""
    def __init__(self, labels: list[str], add_other: bool = True) -> None:
        """Initializer for label sets.

        Args:
            labels: list of class label names.
            add_other: If true, add a other class.
        """
        self._add_other = add_other
        self._labels = labels
        self._other_label = "<O>"
        assert self._other_label not in self._labels, f"{self._other_label} cannot be a label"
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
        return self[self._other_label]

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
    """Label set for part-of-speech tags"""
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


parse_labelset = LabelSet(list("SNCDE"))
pos_labelset = PosLabelSet(["NOUN", "VERB", "ADJECTIVE", "ADVERB"])
ner_labelset = LabelSet(["PERSON", "ORG", "GPE", "LOC"])


class Mention:
    """Mention objects represent mentions with head information. 
    It contains three integers: start token index, end token index, and head token index of the mention. 
    The end token index is inclusive. start <= head <= end.
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
    """CorefDocument objects represent coreference-annotated and parsed movie script. 
    It contains the following attributes: movie, rater, tokens, parse tags, part-of-speech tags, named entity tags,
    speaker tags, sentence offsets, and clusters. 
    Clusters is a dictionary of character names to set of Mention objects.
    """
    def __init__(self, json: dict[str, any] = None) -> None:
        self.movie: str
        self.rater: str
        self.token: list[str]
        self.parse: list[str]
        self.pos: list[str]
        self.ner: list[str]
        self.speaker: list[str]
        self.sentence_offsets: list[list[int]]
        self.clusters: dict[str, set[Mention]]

        # The following fields are derived from the 'parse', 'pos', 'ner', and 'token' fields
        # They hold redundant information but are included for convenience in the character recognition modeling
        self.parse_ids: list[int]
        self.pos_ids: list[int]
        self.ner_ids: list[int]
        self.is_pronoun: list[bool]
        self.is_punctuation: list[bool]

        # The following fields are derived from the 'clusters' field and hold redundant information
        # These are included for convenience
        self.word_cluster_ids: list[int]
        self.word_head_ids: list[int]

        # The following fields are filled during tokenization and corpus preparation stage
        self.subword_ids: list[int]
        self.word_to_subword_offset: list[list[int]]
        self.subword_dataloader: DataLoader

        # This field is fillen when the document is split from a larger document
        self.offset: tuple[int, int]

        if json is not None:
            self.movie = json["movie"]
            self.rater = json["rater"]
            self.token = json["token"]
            self.parse = json["parse"]
            self.pos = json["pos"]
            self.ner = json["ner"]
            self.speaker = json["speaker"]
            self.sentence_offsets = json["sent_offset"]
            self.clusters = {}
            for character, mentions in json["clusters"].items():
                mentions = set([Mention(*x) for x in mentions])
                self.clusters[character] = mentions

            # Filling in the derived fields
            self.parse_ids = [parse_labelset[x] for x in self.parse]
            self.pos_ids = [pos_labelset[x] for x in self.pos]
            self.ner_ids = [ner_labelset[x] for x in self.ner]
            self.is_pronoun = [t.lower() in pronouns for t in self.token]
            self.is_punctuation = [t in punctuation for t in self.token]
            self.word_cluster_ids = np.zeros(len(self.token), dtype=int).tolist()
            self.word_head_ids = np.zeros(len(self.token), dtype=int).tolist()
            for i, (_, mentions) in enumerate(self.clusters.items()):
                for mention in mentions:
                    if len(mentions) > 1:
                        self.word_cluster_ids[mention.head] = i + 1
                    self.word_head_ids[mention.head] = 1

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
                mention_text = " ".join(self.token[mention.begin: mention.end + 1])
                mention_head = self.token[mention.head]
                mention_texts.append(f"{mention_text} ({mention_head})")
            n_rows = math.ceil(len(mention_texts)/3)
            for i in range(n_rows):
                row_mention_texts = mention_texts[i * 3: (i + 1) * 3]
                row_desc = "     ".join([f"{mention_text:25s}" for mention_text in row_mention_texts])
                desc += row_desc + "\n"
            desc += "\n"
        return desc


class CorefCorpus:
    """CorefCorpus is a list of CorefDocuments."""
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
    
    def append(self, doc: CorefDocument):
        self.documents.append(doc)
    
    def __add__(self, other: "CorefCorpus") -> "CorefCorpus":
        corpus = CorefCorpus()
        corpus.documents = self.documents + other.documents
        return corpus


class GraphNode:
    """Graph used for DFS"""
    def __init__(self, id_: int):
        self.id = id_
        self.neighbors: set[GraphNode] = set()
        self.visited = False

    def link(self, other: "GraphNode"):
        self.neighbors.add(other)
        other.neighbors.add(self)

    def __repr__(self) -> str:
        return str(self.id)


class CorefResult:
    """Output of running the coreference model"""
    def __init__(self) -> None:
        self.coref_loss: torch.Tensor
        self.character_loss: torch.Tensor
        self.span_loss: torch.Tensor
        self.character_scores: torch.Tensor
        self.coref_scores: torch.Tensor
        self.top_indices: torch.Tensor
        self.representative_mentions_embedding: torch.Tensor
        self.representative_mentions_position: torch.Tensor
        self.representative_mentions_character_scores: torch.Tensor
        self.head2span: dict[int, tuple[int, int, float]] = {}
        self.predicted_character_heads: np.ndarray
        self.predicted_word_clusters: list[set[int]] = []
        self.predicted_span_clusters: list[set[tuple[int, int]]] = []


class Metric:
    """General metric class for precision, recall, and F1"""
    def __init__(self, recall: float = None, precision: float = None) -> None:
        if recall is not None and precision is not None:
            self.recall = float(100*recall)
            self.precision = float(100*precision)
            self.f1 = 2 * self.precision * self.recall / (1e-23 + self.precision + self.recall)
        else:
            self.recall = self.precision = self.f1 = None
    
    def __repr__(self) -> str:
        return (f"p={self.precision:.1f} r={self.recall:.1f} f1={self.f1:.1f}")
    
    def todict(self) -> dict[str, int]:
        return dict(precision=round(self.precision, 3), recall=round(self.recall, 3), f1=round(self.f1, 3))

class CorefMetric:
    """Metric for coreference resolution"""
    def __init__(self):
        self.muc = Metric()
        self.bcub = Metric()
        self.ceafe = Metric()
        self.lea = Metric()

    @property
    def conll_f1(self) -> float:
        return (self.muc.f1 + self.bcub.f1 + self.ceafe.f1)/3

    def todict(self) -> dict[str, dict[str, int]]:
        return dict(muc=self.muc.todict(), bcub=self.bcub.todict(), ceafe=self.ceafe.todict(), lea=self.lea.todict())

class MovieCorefMetric:
    """Metric for coreference resolution and character head prediction"""
    def __init__(self):
        self.word_coref = CorefMetric()
        self.span_coref = CorefMetric()
        self.character = Metric()

    @property
    def word_conll_score(self) -> float:
        return self.word_coref.conll_f1

    @property
    def span_conll_score(self) -> float:
        return self.span_coref.conll_f1
    
    @property
    def word_lea_score(self) -> float:
        return self.word_coref.lea.f1

    @property
    def span_lea_score(self) -> float:
        return self.span_coref.lea.f1
    
    def __repr__(self) -> str:
        return f"Word={self.word_lea_score:.1f}, Span={self.span_lea_score:.1f}, Character={self.character.f1:.1f}"
    
    def todict(self) -> dict[str, dict[str, dict[str, int]] | dict[str, int]]:
        return dict(word=self.word_coref.todict(), span=self.span_coref.todict(), character=self.character.todict())