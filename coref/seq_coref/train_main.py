"""Train a longformer coreference model on already saved training tensors which
were created from coreference corpus. The training terminates when performance
no longer improves on the development set.
"""

from mica_text_coref.coref.seq_coref.models import coref_longformer
from mica_text_coref.coref.seq_coref.data import data
from mica_text_coref.coref.seq_coref.data import data_util
from mica_text_coref.coref.seq_coref.training import train
from mica_text_coref.coref.seq_coref.utils import util

from absl import app
from absl import flags
from absl import logging
import getpass
import os
import time
import torch

FLAGS = flags.FLAGS
flags.DEFINE_string("conll_directory",
    default="/home/sbaruah_usc_edu/mica_text_coref/data/conll-2012/gold",
    help="Directory containing English conll gold jsonlines files")
flags.DEFINE_string("tensors_directory", default=None,
    help="Directory containing English coreference tensors",
    required=True)
flags.DEFINE_bool("use_gpu", default=True, help="Set to use gpu")
flags.DEFINE_bool("use_data_parallel", default=True,
    help="Set to use data parallelism")
flags.DEFINE_integer("train_batch_size", default=20,
    help="Batch size to use in training")
flags.DEFINE_integer("infer_batch_size", default=20,
    help="Batch size to use in inference")
flags.DEFINE_float("learning_rate", default=1e-5, help="learning rate")
flags.DEFINE_float("weight_decay", default=1e-3,
    help="L2 regularization coefficient")
flags.DEFINE_integer("patience", default=3,
    help=("Number of epochs to wait until performance improves on dev set "
        "before early stopping"))
flags.DEFINE_integer("warmup", default=50,
    help="Number of warmup steps in a linear schedule of the learning rate")
flags.DEFINE_bool("use_official_evaluation", default=True,
    help="Set to use the official perl scorer script in evaluation")
flags.DEFINE_string("official_scorer_script",
    default=("/home/sbaruah_usc_edu/mica_text_coref/coref/seq_coref/"
        "scorer/v8.01/scorer.pl"),
    help="Path to the official perl scorer script")
flags.DEFINE_bool("evaluate_against_original", default=False,
    help=("Set to evaluate against the original coreferece corpus "
        "(before removing overlaps)"))
flags.DEFINE_integer("print_n_batches", default=20,
    help="Number of batches after which to print to stdout")
flags.DEFINE_integer("max_epochs", default=10,
    help="Maximum number of epochs to train")
flags.DEFINE_float("max_grad_norm", default=0.1,
    help="Maximum norm of gradients to which they get clipped")
flags.DEFINE_bool("evaluate_on_train", default=True,
    help="Set to also evaluate on the training dataset")
flags.DEFINE_bool("use_large_longformer", default=False,
    help="Set to use the large version of the LongFormer transformer")
flags.DEFINE_bool("use_dynamic_loading", default=True,
    help=("Set to dynamically load tensors onto GPU at each batch, otherwise"
            " load the full dataset of tensors before training"))
flags.DEFINE_bool("save_model", default=True,
    help="Set to save model's weights")
flags.DEFINE_bool("save_model_after_every_epoch", default=True,
    help=("Has no effect if save_model is False. Otherwise if save_model is"
            " True, set save_model_after_every_epoch to True to save model's"
            " weights after every epoch. If save_model is True but"
            " save_model_after_every_epoch is False, only save the best "
            "model's weights"))
flags.DEFINE_bool("save_predictions", default=True,
    help="Save the groundtruth, prediction, doc ids, and attn tensors")
flags.DEFINE_bool("save_predictions_after_every_epoch", default=True,
    help=("Similar to save_model_after_every_epoch, if"
            " save_predictions_after_every_epoch is True (and save_predictions"
            " is True), save the groundtruth, predictions, doc ids, and attn"
            " tensors after every epoch"))
flags.DEFINE_string("save_directory",
    default="/home/sbaruah_usc_edu/mica_text_coref/data/results",
    help="Directory to which model's weights and predictions will be saved")
flags.DEFINE_bool("evaluate_seq", default=True, help=(
    "Only set this if you are evaluating sequences, not sets of mentions. Set"
    " this to do a simple per-token evaluation."))

def train_main():
    for module_name, flag_items in FLAGS.flags_by_module_dict().items():
        if os.path.join(os.getcwd(), module_name) == __file__:
            for flag_item in flag_items:
                logging.info(f"FLAGS: {flag_item.name:<25s} = "
                                f"{flag_item._value}")

    user = getpass.getuser()
    n_gpus = torch.cuda.device_count()
    use_gpu = n_gpus > 0 and FLAGS.use_gpu
    use_dynamic_loading = use_gpu and FLAGS.use_dynamic_loading
    load_corpora = FLAGS.evaluate_against_original or (
        FLAGS.use_official_evaluation)
    devices = list(range(n_gpus))
    device = "cuda:0"

    start_time = time.time()
    logging.info("Intializing CorefLongformer Model...")
    model = coref_longformer.CorefLongformerModel(
        use_large=FLAGS.use_large_longformer)
    time_taken = util.convert_float_seconds_to_time_string(
        time.time() - start_time)
    logging.info(f"...Initialization done. Time taken = {time_taken}")

    start_time = time.time()
    if use_dynamic_loading:
        logging.info("Using dynamic loading, so loading tensors onto CPU.")
        logging.info("Loading train tensors onto CPU...")
        train_dataset = data_util.load_tensors(
            os.path.join(FLAGS.tensors_directory, "train"), device="cpu")
        logging.info("Loading dev tensors onto cpu...")
        dev_dataset = data_util.load_tensors(
            os.path.join(FLAGS.tensors_directory, "dev"), device="cpu")
    else:
        logging.info("Not using dynamic loading, so loading tensors onto GPU.")
        logging.info(f"Loading train tensors onto {device}...")
        train_dataset = data_util.load_tensors(
            os.path.join(FLAGS.tensors_directory, "train"), device=device)
        logging.info(f"Loading dev tensors onto {device}...")
        dev_dataset = data_util.load_tensors(
            os.path.join(FLAGS.tensors_directory, "dev"), device=device)
    time_taken = util.convert_float_seconds_to_time_string(
        time.time() - start_time)
    logging.info(f"...Loading done. Time taken = {time_taken}")
    util.print_gpu_usage(user, devices)

    longformer_train_corpus = None
    longformer_dev_corpus = None

    if load_corpora:
        logging.info("Loading corpora.")
        train_jsonlines_path = os.path.join(FLAGS.conll_directory, 
            "train.english.jsonlines")
        start_time = time.time()
        logging.info(f"Loading train corpus...")
        train_corpus = data.CorefCorpus(train_jsonlines_path,
            use_ascii_transliteration=True)
        time_taken = util.convert_float_seconds_to_time_string(
            time.time() - start_time)
        logging.info(f"...Loading done. Time taken = {time_taken}")

        start_time = time.time()
        logging.info("Re-tokenizing train corpus using longformer tokenizer...")
        longformer_train_corpus = data_util.remap_spans_document_level(
            train_corpus, model.tokenizer.tokenize)
        time_taken = util.convert_float_seconds_to_time_string(
            time.time() - start_time)
        logging.info("...Re-tokenization done. Time taken = {time_taken}")

        dev_jsonlines_path = os.path.join(FLAGS.conll_directory, 
            "dev.english.jsonlines")
        start_time = time.time()
        logging.info("Loading dev corpus...")
        dev_corpus = data.CorefCorpus(dev_jsonlines_path,
            use_ascii_transliteration=True)
        time_taken = util.convert_float_seconds_to_time_string(
            time.time() - start_time)
        logging.info(f"...Loading done. Time taken = {time_taken}")

        start_time = time.time()
        logging.info("Re-tokenizing dev corpus using longformer tokenizer...")
        longformer_dev_corpus = data_util.remap_spans_document_level(
            dev_corpus, model.tokenizer.tokenize)
        time_taken = util.convert_float_seconds_to_time_string(
            time.time() - start_time)
        logging.info(f"... Re-tokenization done. Time taken = {time_taken}")
    
    train.train(model,
        train_dataset,
        dev_dataset,
        longformer_train_corpus,
        longformer_dev_corpus,
        use_gpu=use_gpu,
        use_data_parallel=FLAGS.use_data_parallel,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.infer_batch_size,
        learning_rate=FLAGS.learning_rate,
        weight_decay=FLAGS.weight_decay,
        print_n_batches=FLAGS.print_n_batches,
        max_epochs=FLAGS.max_epochs,
        max_grad_norm=FLAGS.max_grad_norm,
        patience=FLAGS.patience,
        n_warmup_steps=FLAGS.warmup,
        use_official_evaluation=FLAGS.use_official_evaluation,
        official_scorer=FLAGS.official_scorer_script,
        evaluate_on_train=FLAGS.evaluate_on_train,
        use_dynamic_loading=FLAGS.use_dynamic_loading,
        evaluate_against_original=FLAGS.evaluate_against_original,
        save_model=FLAGS.save_model,
        save_model_after_every_epoch=FLAGS.save_model_after_every_epoch,
        save_predictions=FLAGS.save_predictions,
        save_predictions_after_every_epoch=(
            FLAGS.save_predictions_after_every_epoch),
        save_directory=FLAGS.save_directory,
        evaluate_seq=FLAGS.evaluate_seq
        )

def main(argv):
    logging.get_absl_handler().use_absl_log_file()
    train_main()

if __name__=="__main__":
    app.run(main)