"""Main entry point for baseline calculation of full-length movie scripts"""
from movie_coref import baseline

from absl import app
from absl import flags
import os
import numpy as np

FLAGS = flags.FLAGS
def data_path(path): return os.path.join(os.environ["DATA_DIR"], "mica_text_coref", path)
def code_path(path): return os.path.join(os.environ["PROJ_DIR"], "mica_text_coref", path)

# Directories and Files
flags.DEFINE_string("input_dir", default=data_path("movie_coref/results"),
                    help="Directory containing the preprocessed jsonlines.")
flags.DEFINE_string("output_dir", default=data_path("movie_coref/results/coreference/baselines"),
                    help="Directory to which the baseline predictions will be saved.")
flags.DEFINE_string("weights", default=data_path("word_level_coref/data/roberta_(e20_2021.05.02_01.16)_release.pt"),
                    help="Weights of word-level roberta coreference model.")
flags.DEFINE_string("config", default=data_path("word_level_coref/config.toml"),
                    help="Config file of word-level roberta coreference model.")
flags.DEFINE_string("reference_scorer", default=code_path("coref/movie_coref/scorer/v8.01/scorer.pl"),
                    help="Path to conll coreference scorer.")
flags.DEFINE_enum("genre", default="wb", enum_values=["bc", "bn", "mz", "nw", "pt", "tc", "wb"],
                  help="Genre to use for word-level roberta coreference model.")
flags.DEFINE_integer("batch_size", default=16, help=("Batch size to use for antecedent coreference scoring in "
                                                     "word-level roberta coreference model."))
flags.DEFINE_enum("preprocess", default="none", enum_values=["addsays", "nocharacters", "none"],
                  help="Type of script preprocessing.")
flags.DEFINE_integer("split_len", default=5120, help="Number of words of the smaller screenplays.")
flags.DEFINE_integer("overlap_len", default=512, help="Number of overlapping words between the smaller screenplays")
flags.DEFINE_bool("lea", default=False, help="If true, calculate lea, else calculate conll (muc, bcub, ceafe)")
flags.DEFINE_bool("use_reference_scorer", default=True, help="If true, use reference scorer for conll calculation")
flags.DEFINE_bool("overwrite", default=False, help="Overwrite predictions.")
flags.DEFINE_bool("calc_results", default=False, help="Calculate results.")

def main(argv):
    # Exit if extra command-line args are given
    if len(argv) > 1:
        print("Extra command-line arguments.")
        return

    # Print preprocess, genre, split len, and overlap len
    print(f"preprocess={FLAGS.preprocess} genre={FLAGS.genre} split_len={FLAGS.split_len} "
          f"overlap_len={FLAGS.overlap_len} ")
    
    # Find input file, output file, and result file
    subdir = "regular" if FLAGS.preprocess == "none" else FLAGS.preprocess
    metric_str = "lea" if FLAGS.lea else "conll"
    setting = f"preprocess_{FLAGS.preprocess}.genre_{FLAGS.genre}.split_{FLAGS.split_len}.overlap_{FLAGS.overlap_len}"
    input_file = os.path.join(FLAGS.input_dir, subdir, "train_wl.jsonlines")
    output_file = os.path.join(FLAGS.output_dir, f"{setting}.train_wl")
    result_file = os.path.join(FLAGS.output_dir, f"{setting}.{metric_str}.baseline.tsv")
    
    # # Exit if result file already present
    # if os.path.exists(result_file):
    #     with open(result_file, "r") as fr:
    #         n_lines = len(fr.readlines())
    #     if n_lines > 1:
    #         print("already computed!")
    #         return

    # Evaluate
    result = baseline.wl_evaluate(input_file, output_file, FLAGS.reference_scorer, FLAGS.config, FLAGS.weights,
                                  FLAGS.batch_size, FLAGS.preprocess, FLAGS.genre, FLAGS.split_len, FLAGS.overlap_len,
                                  FLAGS.use_reference_scorer, FLAGS.calc_results, FLAGS.overwrite, FLAGS.lea)
    
    # Write results
    if FLAGS.calc_results:
        with open(result_file, "w") as fw:
            fw.write("preprocess\tgenre\tsplit_len\toverlap_len\tmerge_strategy\tmerge_speakers\tentity\t"
                     "remove_gold_singletons\tprovide_gold_mentions\tmovie\tmetric\tP\tR\tF\n")
            if result is not None:
                for (merge_strategy, merge_speakers, entity, remove_gold_singletons,
                        provide_gold_mentions), setting_dict in result.items():
                    for movie, movie_dict in setting_dict.items():
                        for metric, metric_data in movie_dict.items():
                            scores = np.array([metric_data.precision, metric_data.recall, metric_data.f1])
                            scores = np.round(scores, 6).astype(str).tolist()
                            fw.write("\t".join([FLAGS.preprocess, FLAGS.genre, str(FLAGS.split_len),
                                                str(FLAGS.overlap_len), str(merge_strategy), str(merge_speakers),
                                                entity, str(remove_gold_singletons), str(provide_gold_mentions),
                                                movie, metric] + scores))
                            fw.write("\n")

if __name__ == '__main__':
    app.run(main)