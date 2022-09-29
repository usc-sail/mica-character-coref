"""Main function to print movie script with coreference chains.
"""

from mica_text_coref.coref.movie_coref import data

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "jsonlines", default=None,
    help="Jsonlines file of coreference-annotated movie scripts.",
    required=True)
flags.DEFINE_enum(
    "movie", default=None,
    enum_values=["avengers_endgame","dead_poets_society","john_wick",
                 "prestige","quiet_place","zootopia","shawshank","bourne",
                 "basterds"],
    help="Movie to print.", required=True)

def print(argv):
    corpus = data.CorefCorpus(FLAGS.jsonlines)
    movie = [document for document in corpus.documents
                      if document.movie == FLAGS.movie][0]
    print(movie.movie)

if __name__=="__main__":
    app.run(print)