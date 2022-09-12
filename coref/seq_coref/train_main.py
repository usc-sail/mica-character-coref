"""Train a longformer coreference model on already saved training tensors which
were created from coreference corpus. The training terminates when performance
no longer improves on the development set.
"""

from mica_text_coref.coref.seq_coref import coref_longformer
from mica_text_coref.coref.seq_coref import data
from mica_text_coref.coref.seq_coref import data_util
from mica_text_coref.coref.seq_coref import train
from mica_text_coref.coref.seq_coref import util

from absl import app
from absl import flags
import os
import re
import time

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
flags.DEFINE_bool("use_large_longformer", default=False,
    help="Set to use the large version of the LongFormer transformer")
flags.DEFINE_bool("use_dynamic_loading", default=False,
    help=("Set to dynamically load tensors onto GPU at each batch, otherwise"
            " load the full dataset of tensors before training"))

flags.register_validator("device",
    lambda v: re.match(r"^(cpu)|(cuda:\d+)$", v) is not None,
    message="device should be cpu or cuda:{gpu_index}", flag_values=FLAGS)

def train_main():
    for module_name, flag_items in FLAGS.flags_by_module_dict().items():
        if os.path.join(os.getcwd(), module_name) == __file__:
            for flag_item in flag_items:
                print(f"{flag_item.name:<25s} = {flag_item._value}")
    print("\n")
    
    user = os.getlogin()
    device_index = -1 if FLAGS.device == "cpu" else (
        int(FLAGS.device.lstrip("cuda:")))

    start_time = time.time()
    print("Intializing CorefLongformer Model...\n")
    model = coref_longformer.CorefLongformerModel(
        use_large=FLAGS.use_large_longformer)
    time_taken = util.convert_float_seconds_to_time_string(
        time.time() - start_time)
    print(f"\nInitialization done. Time taken = {time_taken}\n\n")

    start_time = time.time()
    if FLAGS.use_dynamic_loading:
        print("Loading train tensors onto cpu...", end="")
        train_dataset = data_util.load_tensors(
            os.path.join(FLAGS.tensors_directory, "train"), device="cpu")
    else:
        print(f"Loading train tensors onto {FLAGS.device}...", end="")
        train_dataset = data_util.load_tensors(
            os.path.join(FLAGS.tensors_directory, "train"), FLAGS.device)
    time_taken = util.convert_float_seconds_to_time_string(
        time.time() - start_time)
    print(f"done. Time taken = {time_taken}")
    if device_index > -1:
        memory_consumed, memory_available = util.get_gpu_usage(user, 
            device_index)
        print(f"GPU memory consumed = {memory_consumed},"
                f" availabe = {memory_available}")
    print()

    start_time = time.time()
    if FLAGS.use_dynamic_loading:
        print("Loading dev tensors onto cpu...", end="")
        dev_dataset = data_util.load_tensors(
            os.path.join(FLAGS.tensors_directory, "dev"), device="cpu")
    else:
        print(f"Loading dev tensors onto {FLAGS.device}...", end="")
        dev_dataset = data_util.load_tensors(
            os.path.join(FLAGS.tensors_directory, "dev"), FLAGS.device)
    time_taken = util.convert_float_seconds_to_time_string(
        time.time() - start_time)
    print(f"done. Time taken = {time_taken}")
    if device_index > -1:
        memory_consumed, memory_available = util.get_gpu_usage(user, 
            device_index)
        print(f"GPU memory consumed = {memory_consumed},"
                f" availabel = {memory_available}")
    print()

    longformer_train_corpus = None
    longformer_dev_corpus = None

    if FLAGS.evaluate_against_original:
        train_jsonlines_path = os.path.join(FLAGS.conll_directory, 
            "train.english.jsonlines")
        start_time = time.time()
        print(f"Loading train corpus...", end="")
        train_corpus = data.CorefCorpus(train_jsonlines_path,
            use_ascii_transliteration=True)
        time_taken = util.convert_float_seconds_to_time_string(
            time.time() - start_time)
        print(f"done. Time taken = {time_taken}")

        start_time = time.time()
        print(f"Re-tokenizing train corpus using longformer tokenizer...",
            end="")
        longformer_train_corpus = data_util.remap_spans_document_level(
            train_corpus, model.tokenizer.tokenize)
        time_taken = util.convert_float_seconds_to_time_string(
            time.time() - start_time)
        print(f"done. Time taken = {time_taken}\n")

        dev_jsonlines_path = os.path.join(FLAGS.conll_directory, 
            "dev.english.jsonlines")
        start_time = time.time()
        print(f"Loading dev corpus...", end="")
        dev_corpus = data.CorefCorpus(dev_jsonlines_path,
            use_ascii_transliteration=True)
        time_taken = util.convert_float_seconds_to_time_string(
            time.time() - start_time)
        print(f"done. Time taken = {time_taken}")

        start_time = time.time()
        print(f"Re-tokenizing dev corpus using longformer tokenizer...",
            end="")
        longformer_dev_corpus = data_util.remap_spans_document_level(
            dev_corpus, model.tokenizer.tokenize)
        time_taken = util.convert_float_seconds_to_time_string(
            time.time() - start_time)
        print(f"done. Time taken = {time_taken}")
    
    train.train(model,
        train_dataset,
        dev_dataset,
        longformer_train_corpus,
        longformer_dev_corpus,
        device=FLAGS.device,
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
        use_dynamic_loading=FLAGS.use_dynamic_loading)

def main(argv):
    train_main()

if __name__=="__main__":
    app.run(main)