"""Main function to preprocess screenplay text, screenplay parse csv, and screenplay coreference csv to jsonlines, 
conll, and jsonlines for word-level coreference modeling.
"""

from movie_coref import preprocess

from absl import app
from absl import flags
import os

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", default="data", help="Data directory containing screenplays and parsed "
                    "screenplays. Screenplays should end in '_script.txt' and its parsed file should end in "
                    "'_parse.txt' with the same prefix.")
flags.DEFINE_bool("gold", default=False, help="Set if data directory contains annotations (only use it for "
                  "model training)")
flags.DEFINE_integer("device_id", default=-1, help="GPU device index. -1 for CPU.")

def main(argv):
    if len(argv) > 1:
        print("too many command-line arguments")
        return
    data_dir = FLAGS.data_dir
    gold = FLAGS.gold
    device_id = FLAGS.device_id
    parse = os.path.join(data_dir, "parse.csv")
    movie_raters = os.path.join(data_dir, "movies.txt")
    screenplays = os.path.join(data_dir, "screenplay")
    annotations = os.path.join(data_dir, "labels")
    output = data_dir
    if gold:
        preprocess.convert_screenplay_and_coreference_annotation_to_json(parse, movie_raters, screenplays, annotations, 
                                                                         output, spacy_gpu_device=device_id)
    else:
        script_names = set()
        parse_names = set()
        for file in os.listdir(data_dir):
            if file.endswith("_script.txt"):
                script_names.add(file[:-11])
            elif file.endswith("_parse.txt"):
                parse_names.add(file[:-10])
        names = script_names.intersection(parse_names)
        if names:
            script_files = [os.path.join(data_dir, f"{name}_script.txt") for name in names]
            parse_files = [os.path.join(data_dir, f"{name}_parse.txt") for name in names]
            output_file = os.path.join(data_dir, f"coref_input.jsonl")
            preprocess.preprocess_scripts(script_files, parse_files, output_file, gpu_device=device_id)

if __name__=="__main__":
    app.run(main)