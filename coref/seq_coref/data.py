"""Contains class definitions of data structures
"""

from enum import Enum
import jsonlines


class MentionPairRelationship(Enum):
    DISJOINT = 0
    SUBSPAN = 1
    INTERSECT = 2
    EQUAL = 3


class Mention:
    """Data structure of a mention. It defines a text span's indexes.
    """

    def __init__(self, begin:int, end:int) -> None:
        self.begin = begin
        self.end = end

    def __eq__(self, __o:"Mention") -> bool:
        return self.begin == __o.begin and self.end == __o.end

    def __lt__(self, __o:"Mention") -> bool:
        return self.begin < __o.begin or self.end < __o.end
    
    def __le__(self, __o:"Mention") -> bool:
        return self.begin <= __o.begin or self.end <= __o.end
    
    def __ne__(self, __o:"Mention") -> bool:
        return not self == __o
    
    def __hash__(self) -> int:
        return hash((self.begin, self.end))
    
    def __repr__(self) -> str:
        return f"({self.begin},{self.end})"
    
    def __len__(self) -> int:
        return self.end - self.begin + 1


class CorefDocument:
    """Data structure of a text document, annotated with coreference relations.
    It contains the document key, sentences, speakers, and clusters.
    Sentences is a list of list of strings. Speakers is also a list of list of
    strings, of the same dimensions as sentences. Clusters is a list of set of
    data.Mention objects, each set representing a coreference chain.
    """

    def __init__(self, json: dict[any]) -> None:
        self.doc_key: str = json["doc_key"]
        self.sentences: list[list[str]] = json["sentences"]
        self.speakers: list[list[str]] = json["speakers"]
        self.named_entities: dict[Mention, str] = {}
        self.constituents: dict[Mention, str] = {}
        self.clusters: list[set[Mention]] = []
        for annotated_mention in json["ner"]:
            mention = Mention(annotated_mention[0], annotated_mention[1])
            self.named_entities[mention] = annotated_mention[2]
        for annotated_mention in json["constituents"]:
            mention = Mention(annotated_mention[0], annotated_mention[1])
            self.constituents[mention] = annotated_mention[2]
        for cluster in json["clusters"]:
            cluster_set = set() 
            for mention in cluster:
                mention = Mention(mention[0], mention[1])
                cluster_set.add(mention)
            self.clusters.append(cluster_set)


class CorefCorpus:
    """
    Data structure of a list of coreference-annotated text documents.
    """

    def __init__(self, jsonlines_file: str | None = None) -> None:
        self.documents: list[CorefDocument] = []
        if jsonlines_file is not None:
            with jsonlines.open(jsonlines_file) as reader:
                for document in reader:
                    document = CorefDocument(document)
                    self.documents.append(document)

    def __add__(self, other: "CorefCorpus") -> "CorefCorpus":
        combined = CorefCorpus()
        combined.documents = self.documents + other.documents
        return combined
    
    def __len__(self) -> int:
        return len(self.documents)