"""Test preprocess
"""

from mica_text_coref.coref.movie_coref import data
from mica_text_coref.coref.movie_coref import preprocess

import collections
import os
import pandas as pd
import re
import shutil
import unidecode
import unittest

class TestPreprocess(unittest.TestCase):
    """Testing preprocess library.
    """
    proj_dir = os.getcwd()
    output_dir = os.path.join(proj_dir, "data/temp/test_preprocess")
    data_dir = os.path.join(proj_dir, "data/movie_coref")
    screenplay_parse_file = os.path.join(data_dir, "parse.csv")
    movie_and_raters_file = os.path.join(data_dir, "movies.txt")
    screenplays_dir = os.path.join(data_dir, "screenplay")
    annotations_dir = os.path.join(data_dir, "csv")

    @classmethod
    def setUpClass(cls):
        os.makedirs(cls.output_dir, exist_ok=False)
        preprocess.convert_screenplay_and_coreference_annotation_to_json(
            cls.screenplay_parse_file, cls.movie_and_raters_file,
            cls.screenplays_dir, cls.annotations_dir, cls.output_dir,
            spacy_gpu_device=0)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.output_dir)

    def __test_sentence_offsets(self, corpus: data.CorefCorpus):
        for document in corpus:
            tokens = document.token
            sent_offsets = document.sentence_offsets
            if len(tokens) > 0:
                self.assertEqual(sent_offsets[0][0], 0,
                                 "First sentence should start from 0th token")
                for i in range(len(sent_offsets) - 1):
                    self.assertEqual(
                        sent_offsets[i][1] + 1, sent_offsets[i + 1][0],
                        (f"{i + 1}th sentence should start after end of {i}th "
                         "sentence"))
                self.assertEqual(sent_offsets[-1][1], len(tokens) - 1, 
                                 "Last sentence should end in last token")
    
    def __normalize_text(self, text: str) -> str:
        text = re.sub("\s+", " ", text)
        text = text.rstrip(".")
        text = unidecode.unidecode(text, errors="strict")
        return text

    def __test_clusters(self, corpus: data.CorefCorpus, annotations_dir: str,
                        screenplays_dir: str):
        for document in corpus:
            movie, tokens, clusters = (document.movie, document.token,
                                       document.clusters)
            annotations_file = os.path.join(annotations_dir, f"{movie}.csv")
            script_file = os.path.join(screenplays_dir, f"{movie}.txt")
            document_ch2cl = collections.defaultdict(list)
            script_ch2cl = collections.defaultdict(list)

            for character, mentions in clusters.items():
                for mention in mentions:
                    mention_text = " ".join(tokens[mention.begin:
                                                   mention.end + 1])
                    mention_text = self.__normalize_text(mention_text)
                    document_ch2cl[character].append(mention_text)

            with open(script_file, encoding="utf-8") as fr:
                script = fr.read()
            annotations_df = pd.read_csv(annotations_file, index_col=None)
            for _, row in annotations_df.iterrows():
                character = row["entityLabel"]
                begin, end = int(row["begin"]), int(row["end"])
                mention_text = script[begin: end]
                mention_text = self.__normalize_text(mention_text)
                script_ch2cl[character].append(mention_text)
            
            document_characters = sorted(document_ch2cl.keys())
            script_characters = sorted(script_ch2cl.keys())
            self.assertListEqual(document_characters, script_characters,
                            f"movie={movie}\ndocument-characters="
                            f"{document_characters}\nscript-characters="
                            f"{script_characters}")
    
    def test_regular(self):
        filename = os.path.join(self.output_dir, "regular/movie.jsonlines")
        corpus = data.CorefCorpus(filename)
        self.__test_sentence_offsets(corpus)
        self.__test_clusters(corpus, self.annotations_dir, self.screenplays_dir)
    
    def test_addsays(self):
        filename = os.path.join(self.output_dir, "addsays/movie.jsonlines")
        corpus = data.CorefCorpus(filename)
        self.__test_sentence_offsets(corpus)
        self.__test_clusters(corpus, self.annotations_dir, self.screenplays_dir)

    def test_nocharacters(self):
        filename = os.path.join(self.output_dir, "nocharacters/movie.jsonlines")
        corpus = data.CorefCorpus(filename)
        self.__test_sentence_offsets(corpus)
        self.__test_clusters(corpus, self.annotations_dir, self.screenplays_dir)

if __name__=="__main__":
    unittest.main()