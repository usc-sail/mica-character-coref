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

    def __init__(self, json: dict[any]) -> None:
        self.doc_key: str = json["doc_key"]
        self.sentences: list[list[str]] = json["sentences"]
        self.speakers: list[list[str]] = json["speakers"]
        self.clusters: list[set[Mention]] = []
        for cluster in json["clusters"]:
            cluster_set = set() 
            for mention in cluster:
                mention = Mention(mention[0], mention[1])
                cluster_set.add(mention)
            self.clusters.append(cluster_set)


class CorefCorpus:

    def __init__(self, jsonlines_file) -> None:
        self.documents: list[CorefDocument] = []
        with jsonlines.open(jsonlines_file) as reader:
            for document in reader:
                document = CorefDocument(document)
                self.documents.append(document)