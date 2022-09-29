"""Data structure for movie coreference for fine-tuning worl-level coreference
models
"""

import jsonlines
import math

class Mention:

    def __init__(self, begin: int, end: int, head: int | None) -> None:
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

    def __init__(self, json: dict[str, any]) -> None:
        self.movie: str = json["movie"]
        self.rater: str = json["rater"]
        self.token: list[str] = json["token"]
        self.parse: list[str] = json["parse"]
        self.pos: list[str] = json["pos"]
        self.ner: list[str] = json["ner"]
        self.speaker: list[str] = json["speaker"]
        self.sentence_offsets: list[tuple[int, int]] = json["sent_offset"]
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