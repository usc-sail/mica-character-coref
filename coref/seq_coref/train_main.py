"""Train a longformer coreference model on already saved training tensors which
were created from coreference corpus. The training terminates when performance
no longer improves on the development set.
"""

from mica_text_coref.coref.seq_coref import train

from absl import app
from absl import flags
import os

FLAGS = flags.FLAGS
flags.DEFINE_string("conll_directory",
    default="/home/sbaruah_usc_edu/mica_text_coref/data/conll-2012/gold",
    help="Directory containing English conll gold jsonlines files")
flags.DEFINE_string("tensors_directory",
    default=("/home/sbaruah_usc_edu/mica_text_coref/data/"
            "tensors/longformer_seq_tensors"),
    help="Directory containing English coreference tensors")
flags.DEFINE_string("device", default="cuda:0", help="Device on which to train")
flags.DEFINE_integer("train_batch_size", default=64,
    help="Batch size to use in training")
flags.DEFINE_integer("infer_batch_size", default=64,
    help="Batch size to use in inference")
flags.DEFINE_float("learning_rate", default=1e-5, help="learning rate")
flags.DEFINE_float("weight_decay", default=1e-3,
    help="L2 regularization coefficient")
flags.DEFINE_integer("patience", default=3,
    help=("Number of epochs to wait until performance improves on dev set "
        "before early stopping"))
flags.DEFINE_integer("warmup", default=50,
    help="Number of warmup steps in a linear schedule of the learning rate")
flags.DEFINE_bool("use_official_evaluation", default=False,
    help="Set to use the official perl scorer script in evaluation")
flags.DEFINE_string("official_scorer_script",
    default=("/home/sbaruah_usc_edu/mica_text_coref/coref/seq_coref/"
        "scorer/v8.01/scorer.pl"),
    help="Path to the official perl scorer script")
flags.DEFINE_bool("evaluate_against_original", default=False,
    help=("Set to evaluate against the original coreferece corpus "
        "(before removing overlaps)"))
flags.DEFINE_integer("print_n_batches", default=10,
    help="Number of batches after which to print to stdout")
flags.DEFINE_integer("max_epochs", default=10,
    help="Maximum number of epochs to train")
flags.DEFINE_float("max_grad_norm", default=0.1,
    help="Maximum norm of gradients to which they get clipped")
flags.DEFINE_bool("evaluate_on_train", default=True,
    help="Set to also evaluate on the training dataset")

def train_main():
    for module_name, flag_items in FLAGS.flags_by_module_dict().items():
        if os.path.join(os.getcwd(), module_name) == __file__:
            for flag_item in flag_items:
                print(f"{flag_item.name:<25s} = {flag_item._value}")

def main(argv):
    train_main()

if __name__=="__main__":
    app.run(main)