"""Main entry point for baseline calculation.
"""

from mica_text_coref.coref.movie_coref import baseline

from absl import app
from absl import flags
import os
import sys

FLAGS = flags.FLAGS
def d(path): return os.path.join(os.environ["DATA_DIR"], "mica_text_coref", path)
def c(path): return os.path.join(os.environ["PROJ_DIR"], "mica_text_coref", path)

# Directories and Files
flags.DEFINE_string("input_dir", default=d("movie_coref/results"), help="Directory containing the preprocessed jsonlines.")
flags.DEFINE_string("output_dir", default=d("movie_coref/results/coreference/baselines"), help="Directory to which the baseline predictions will be saved.")
flags.DEFINE_string("wl_weights", default=d("word_level_coref/data/roberta_(e20_2021.05.02_01.16)_release.pt"), help="Weights of word-level roberta coreference model.")
flags.DEFINE_string("wl_config", default=c("coref/word_level_coref/config.toml"), help="Config file of word-level roberta coreference model.")
flags.DEFINE_string("reference_scorer", default=c("coref/movie_coref/scorer/v8.01/scorer.pl"), help="Path to conll reference scorer.")

# Config for Word-Level Roberta Model
flags.DEFINE_enum("wl_genre", default="wb", enum_values=["bc", "bn", "mz", "nw", "pt", "tc", "wb"], help="Genre to use for word-level roberta coreference model.")
flags.DEFINE_integer("wl_batch_size", default=16, help="Batch size to use for antecedent coreference scoring in word-level roberta coreference model.")

# Rules
flags.DEFINE_enum("preprocess", default="none", enum_values=["addsays", "nocharacters", "none"], help="Type of script preprocessing.")
flags.DEFINE_enum("entity", default="all", enum_values=["person", "speaker", "all"], help="Filter entities criterion.")
flags.DEFINE_bool("merge_speakers", default=False, help="Merge clusters by speaker.")
flags.DEFINE_bool("provide_gold_mentions", default=False, help="Provide gold mentions to predictor.")
flags.DEFINE_bool("remove_gold_singletons", default=False, help="Remove singletons from annotations.")
flags.DEFINE_integer("split_len", default=None, help="Number of words of the smaller screenplays. If None, then no splitting occurs.")
flags.DEFINE_integer("overlap_len", default=0, help="Number of overlapping words between the smaller screenplays")
flags.DEFINE_enum("merge_strategy", default="max", enum_values=["none", "max", "min", "before", "after", "average"], help="Merging strategy of the predictions of the split screenplays.")

flags.DEFINE_bool("run_train", default=False, help="Run baseline on training scripts.")
flags.DEFINE_bool("use_reference_scorer", default=True, help="Use reference scorer.")
flags.DEFINE_bool("overwrite", default=False, help="Overwrite predictions.")
flags.DEFINE_bool("calc_results", default=False, help="Calculate results.")
flags.DEFINE_bool("use_gpu", default=False, help="Use cuda:0 gpu if available")

def main(argv):
    # Exit if extra command-line args are given
    if len(argv) > 1: sys.exit("Extra command-line arguments.")

    # Preprocess, Partition, Input File, Output File
    subdir = "regular" if FLAGS.preprocess == "none" else FLAGS.preprocess
    partition = "train" if FLAGS.run_train else "dev"
    split_str = f"split_{FLAGS.split_len}.overlap_{FLAGS.overlap_len}." if FLAGS.split_len is not None else ""
    input_file = os.path.join(FLAGS.input_dir, subdir, f"{partition}_wl.jsonlines")
    output_file = os.path.join(FLAGS.output_dir, f"preprocess_{FLAGS.preprocess}.genre_{FLAGS.wl_genre}.{split_str}{partition}_wl")
    
    # Exit if result file already present
    setting = (f"preprocess_{FLAGS.preprocess}.genre_{FLAGS.wl_genre}.entity_{FLAGS.entity}.merge_speakers_{FLAGS.merge_speakers}.provide_gold_mentions_{FLAGS.provide_gold_mentions}."
            f"remove_gold_singletons_{FLAGS.remove_gold_singletons}.{split_str}merge_strategy_{FLAGS.merge_strategy}")
    result_file = os.path.join(FLAGS.output_dir, f"{partition}.{setting}.baseline.tsv")
    if os.path.exists(result_file):
        print("already computed!")
        return

    # Evaluate
    result = baseline.wl_evaluate(FLAGS.reference_scorer, FLAGS.wl_config, FLAGS.wl_weights, FLAGS.wl_batch_size, FLAGS.wl_genre, input_file, output_file, FLAGS.entity, FLAGS.merge_speakers, 
        FLAGS.provide_gold_mentions, FLAGS.remove_gold_singletons, FLAGS.split_len, FLAGS.overlap_len, FLAGS.merge_strategy, FLAGS.use_reference_scorer, FLAGS.calc_results, 
        FLAGS.overwrite, FLAGS.use_gpu)
    
    # Write results
    if FLAGS.calc_results:
        with open(result_file, "w") as fw:
            fw.write("preprocess\tgenre\tentity\tmerge_speakers\tprovide_gold_mentions\tremove_gold_singletons\tsplit_len\toverlap_len\tmerge_strategy\tmetric\tmovie\tP\tR\tF\n")
            if result is not None:
                for metric, metric_result in result.items():
                    for movie, movie_metric in metric_result.items():
                        fw.write("\t".join([FLAGS.preprocess, FLAGS.wl_genre, FLAGS.entity, str(FLAGS.merge_speakers), str(FLAGS.provide_gold_mentions), str(FLAGS.remove_gold_singletons), 
                            str(FLAGS.split_len), str(FLAGS.overlap_len), FLAGS.merge_strategy, metric, movie] + movie_metric.tolist()))
                        fw.write("\n")

if __name__ == '__main__':
    app.run(main)