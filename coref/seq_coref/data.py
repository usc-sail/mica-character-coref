"""Contains class definitions of data structures
"""

from enum import Enum
import jsonlines
import re
import tqdm
import unidecode

class MentionPairRelationship(Enum):
    DISJOINT = 0
    SUBSPAN = 1
    INTERSECT = 2
    EQUAL = 3


class Mention:
    """Data structure of a mention. It defines a text span's indexes. The begin
    and end indexes are inclusive.
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
    It contains the document key, sentences, speakers, clusters, named entity,
    and constituents. Sentences is a list of list of strings. Speakers is also a
    list of list of strings, of the same dimensions as sentences. Clusters is a
    list of set of data.Mention objects, each set representing a coreference
    chain. Named entities and constituents are dictionaries of data.Mention
    objects to their named entity and constituency tag respectively.
    """

    def __init__(self, json: None | dict[any] = None, 
                use_ascii_transliteration = False, verbose = False) -> None:
        """Set use_ascii_transliteration for English documents only."""
        self.doc_id: int = -1
        self.doc_key: str = ""
        self.sentences: list[list[str]] = []
        self.speakers: list[list[str]] = []
        self.named_entities: dict[Mention, str] = {}
        self.constituents: dict[Mention, str] = {}
        self.clusters: list[set[Mention]] = []

        if json is not None:
            self.doc_key: str = json["doc_key"]
            self.sentences: list[list[str]] = json["sentences"]
            self.speakers: list[list[str]] = json["speakers"]
            new_sentences = []

            for sentence in self.sentences:
                new_sentence = []
                for word in sentence:
                    if word == "-LRB-":
                        word = "("
                    elif word == "-RRB-":
                        word = ")"
                    elif word == "-LSB-":
                        word = "["
                    elif word == "-RSB-":
                        word = "]"
                    elif word == "-LCB-":
                        word = "{"
                    elif word == "-RCB-":
                        word = "}"
                    if use_ascii_transliteration:
                        ascii_word = re.sub(r"\s", "", unidecode.unidecode(
                            word))
                        if len(ascii_word) == 0:
                            ascii_word = "."
                        if verbose and ascii_word != word:
                            print(f"Using '{ascii_word}' instead of '{word}'")
                    else:
                        ascii_word = word
                    new_sentence.append(ascii_word)
                new_sentences.append(new_sentence)
            self.sentences = new_sentences

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

    def __init__(self, jsonlines_file: str | None = None, 
                use_ascii_transliteration = False, verbose = False) -> None:
        """Set use_ascii_transliteration for English corpus only."""
        self.documents: list[CorefDocument] = []
        if jsonlines_file is not None:
            with jsonlines.open(jsonlines_file) as reader:
                if verbose:
                    corpus_reader = tqdm.tqdm(enumerate(reader),
                                             desc="reading jsonlines")
                else:
                    corpus_reader = enumerate(reader)
                for i, document in corpus_reader:
                    document = CorefDocument(
                        document, 
                        use_ascii_transliteration=use_ascii_transliteration, 
                        verbose=verbose)
                    document.doc_id = i
                    self.documents.append(document)

    def __add__(self, other: "CorefCorpus") -> "CorefCorpus":
        combined = CorefCorpus()
        combined.documents = self.documents + other.documents
        for i, document in enumerate(combined.documents):
            document.doc_id = i
        return combined
    
    def get_doc_id_to_doc_key(self) -> dict[int, str]:
        doc_id_to_doc_key: dict[int, str] = {}
        for document in self.documents:
            doc_id_to_doc_key[document.doc_id] = document.doc_key
        return doc_id_to_doc_key

    def get_doc_id_to_sentences(self) -> dict[int, list[list[str]]]:
        doc_id_to_sentences: dict[int, list[list[str]]] = {}
        for document in self.documents:
            doc_id_to_sentences[document.doc_id] = document.sentences
        return doc_id_to_sentences

    def __len__(self) -> int:
        return len(self.documents)