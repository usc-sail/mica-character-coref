"""Train a longformer coreference model on already saved training tensors which
were created from coreference corpus. The training terminates when performance
no longer improves on the development set.
"""

from mica_text_coref.coref.seq_coref import train

from absl import app
from absl import flags
from absl import logging
import os

FLAGS = flags.FLAGS
proj_dir = os.getcwd()

# directories and files
flags.DEFINE_string(
    "conll_dir",
    default=os.path.join(proj_dir, "data/conll-2012/gold"),
    help="Directory containing English conll gold jsonlines files.")
flags.DEFINE_string(
    "tensors_dir",
    default=os.path.join(proj_dir, "data/tensors/longformer_seq_tensors_512"),
    help="Directory containing English coreference tensors.")
flags.DEFINE_string(
    "output_dir", default=os.path.join(proj_dir, "data/results"),
    help="Directory to which model's weights and predictions will be saved")
flags.DEFINE_string("perl_scorer",
    default=(os.path.join(proj_dir, "coref/seq_coref/scorer/v8.01/scorer.pl")),
    help="Path to the official perl scorer script")

# model
flags.DEFINE_bool(
    "use_large_longformer", default=False,
    help="Set to use the large version of the LongFormer transformer")

# training
flags.DEFINE_integer(
    "train_batch_size", default=20, lower_bound=1,
    help="Batch size to use in training.")
flags.DEFINE_float(
    "learning_rate", default=1e-5, lower_bound=0, help="learning rate.")
flags.DEFINE_float(
    "weight_decay", default=1e-3, lower_bound=0,
    help="L2 regularization coefficient.")
flags.DEFINE_integer(
    "patience", default=3, lower_bound=1,
    help=("Number of epochs to wait until performance improves on dev set "
          "before early stopping."))
flags.DEFINE_float(
    "warmup_ratio", default=0.1, lower_bound=0, upper_bound=1,
    help="Fraction of training steps used in warmup.")
flags.DEFINE_integer(
    "print_n_batches", default=20, lower_bound=1,
    help="Number of batches after which to log.")
flags.DEFINE_integer(
    "max_epochs", default=10, lower_bound=1,
    help="Maximum number of epochs to train.")
flags.DEFINE_float(
    "max_grad_norm", default=0.1, lower_bound=0,
    help="Maximum norm of gradients to which they get clipped.")

# evaluation
flags.DEFINE_enum(
    "evaluation_strategy", default="seq", enum_values=["seq", "perl", "scorch"],
    help=("Evaluation strategy to use. Choose seq to evaluate sequences only,"
          " perl to evaluate coreference (MUC, B3, CEAF) using the perl scorer,"
          " and scorch to evaluate coreference using the scorch package."))
flags.DEFINE_bool(
    "evaluate_train", default=True, help="Evaluate on the training dataset.")

# inference
flags.DEFINE_integer(
    "infer_batch_size", default=20, help="Batch size to use in inference.")

# save
flags.DEFINE_bool("save_model", default=True, help="Save model's weights.")
flags.DEFINE_bool(
    "save_predictions", default=True,
    help="Save the groundtruth, prediction, doc ids, and attn tensors.")

def train_main():
    # Log command-line arguments
    for module_name, flag_items in FLAGS.flags_by_module_dict().items():
        if os.path.join(os.getcwd(), module_name) == __file__:
            for flag_item in flag_items:
                logging.info(f"FLAGS: {flag_item.name:<25s} = "
                             f"{flag_item._value}")
    
    # Call train
    train.train(tensors_dir=FLAGS.tensors_dir,
                perl_scorer=FLAGS.perl_scorer,
                output_dir=FLAGS.output_dir,
                large_longformer=FLAGS.use_large_longformer,
                train_batch_size=FLAGS.train_batch_size,
                infer_batch_size=FLAGS.infer_batch_size,
                learning_rate=FLAGS.learning_rate,
                weight_decay=FLAGS.weight_decay,
                print_n_batches=FLAGS.print_n_batches,
                max_epochs=FLAGS.max_epochs,
                max_grad_norm=FLAGS.max_grad_norm,
                patience=FLAGS.patience,
                warmup_ratio=FLAGS.warmup_ratio,
                evaluation_strategy=FLAGS.evaluation_strategy,
                evaluate_train=FLAGS.evaluate_train,
                save_model=FLAGS.save_model,
                save_predictions=FLAGS.save_predictions
                )

def main(argv):
    logging.get_absl_handler().use_absl_log_file()
    train_main()

if __name__=="__main__":
    app.run(main)