"""Find character coreference clusters in movie data jsonlines"""
from movie_coref.movie_coref import MovieCoreference

from absl import flags
from absl import app

FLAGS = flags.FLAGS
flags.DEFINE_string("input_file", default=None, help="Input preprocess movie data jsonlines file.", required=True)
flags.DEFINE_string("weights_file", default=None, help="Trained model weights.", required=True)
flags.DEFINE_integer("subdocument_len", default=5120, help="Subdocument length", lower_bound=512)
flags.DEFINE_integer("overlap_len", default=2048, help="Overlap length (fusion)", lower_bound=256)
flags.DEFINE_integer("repk", default=3, help="Number of representative mentions (hierarchical)", lower_bound=1)
flags.DEFINE_bool("hierarchical", default=False, help="Set for hierarchical inference, otherwise fusion-based "
                  "inference is performed.")

def predict(argv):
    if len(argv) > 1:
        print(f"Extra command-line arguments: {argv}")
        return
    
    movie_coref = MovieCoreference(
        full_length_scripts_file=FLAGS.input_file,
        weights_file=FLAGS.weights_file,
        document_len=FLAGS.subdocument_len,
        overlap_len=FLAGS.overlap_len,
        hierarchical=FLAGS.hierarchical,
        n_representative_mentions=FLAGS.repk,
        save_log=False,
        save_predictions=False,
        save_loss_curve=False
        )
    
    movie_coref.predict()

if __name__=="__main__":
    app.run(predict)