"""Main function to preprocess screenplay text, screenplay parse csv, and
screenplay coreference csv to jsonlines, conll, and jsonlines for word-level
coreference modeling.
"""

from mica_text_coref.coref.movie_coref import preprocess

from absl import app
from absl import flags
import os

FLAGS = flags.FLAGS
proj_dir = os.getcwd()
flags.DEFINE_string("screenplay_parse_csv", default=os.path.join(proj_dir,
    "data/movie_coref/parse.csv"), help="Screenplay parse csv file.")
flags.DEFINE_string("movie_raters_text", default=os.path.join(proj_dir,
    "data/movie_coref/movies.txt"), help="Movie and raters text file.")
flags.DEFINE_string("screenplays_dir", default=os.path.join(proj_dir,
    "data/movie_coref/screenplay"), help="Directory containing screenplay text files.")
flags.DEFINE_string("annotation_dir", default=os.path.join(proj_dir, "data/movie_coref/csv"),
    help="Directory containing annotated csv files.")
flags.DEFINE_string("output_dir", default=os.path.join(proj_dir, "data/movie_coref/results"),
    help="Directory to which the jsonlines and conll files will be saved.")

def main(argv):
    parse = FLAGS.screenplay_parse_csv
    movie_raters = FLAGS.movie_raters_text
    screenplays = FLAGS.screenplays_dir
    annotations = FLAGS.annotation_dir
    output = FLAGS.output_dir
    preprocess.convert_screenplay_and_coreference_annotation_to_json(
        parse, movie_raters, screenplays, annotations, output)

if __name__=="__main__":
    app.run(main)