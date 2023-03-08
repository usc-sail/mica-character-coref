"""Main function to preprocess screenplay text, screenplay parse csv, and screenplay coreference csv to jsonlines, 
conll, and jsonlines for word-level coreference modeling.
"""

from movie_coref import preprocess

from absl import app
from absl import flags
import os

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", default=None, required=True,
                    help="Data directory containing screenplays, parsed screenplays, and annotations.")

def main(argv):
    if len(argv) > 1:
        print("too many command-line arguments")
        return
    data_dir = FLAGS.data_dir
    parse = os.path.join(data_dir, "parse.csv")
    movie_raters = os.path.join(data_dir, "movies.txt")
    screenplays = os.path.join(data_dir, "screenplay")
    annotations = os.path.join(data_dir, "labels")
    output = data_dir
    preprocess.convert_screenplay_and_coreference_annotation_to_json(parse, movie_raters, screenplays, annotations, 
                                                                     output)

if __name__=="__main__":
    app.run(main)