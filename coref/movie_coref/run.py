"""Main entry-point for coreference model training.
"""
from mica_text_coref.coref.movie_coref.coreference_trainer import CoreferenceTrainer

from absl import flags
from absl import app
import datetime
import os
import pytz

FLAGS = flags.FLAGS
proj_dir = os.getcwd()

# Input and Output
flags.DEFINE_string(
    "input_dir", 
    os.path.join(proj_dir, "data/movie_coref/results"),
    "Directory of train and dev jsonlines.")
flags.DEFINE_enum(
    "input_type", "regular", ["regular", "nocharacters", "addsays"], 
    "Type of preprocessing applied to screenplays.")
flags.DEFINE_string(
    "weights_file", 
    os.path.join(proj_dir, "data/word_level_coref/data/"
        "roberta_(e20_2021.05.02_01.16)_release.pt"),
    "Filepath of word-level coreference roberta model's weights.")
flags.DEFINE_string(
    "output_dir", 
    os.path.join(proj_dir, "data/movie_coref/results/coreference"),
    "Directory to save model weights, predictions, evaluation metrics, and "
    "loss curves.")
flags.DEFINE_string(
    "logs_dir", 
    os.path.join(proj_dir, "data/movie_coref/results/coreference/logs"),
    "Directory to save logs.")
flags.DEFINE_bool("save_model", False, "Save model weights.")
flags.DEFINE_bool("save_tensors", False, "Save predictions.")
flags.DEFINE_bool("save_loss_curves", False, "Save loss curves.")
flags.DEFINE_bool("debug", False, "Debug output.")

# Training
flags.DEFINE_bool("freeze_bert", False, help="Freeze RoBerta.")
flags.DEFINE_enum(
    "genre", "wb", ["bc", "bn", "mz", "nw", "pt", "tc", "wb"], "Genre.")
flags.DEFINE_float("bce_weight", 0.5, "Weight of BCE coreference loss.")
flags.DEFINE_float("bert_lr", 1e-5, "Learning rate for the transformer.")
flags.DEFINE_float("lr", 3e-4, "Learning rate for the model.")
flags.DEFINE_float("weight_decay", 1e-3, "Weight decay.")
flags.DEFINE_integer("max_epochs", 10, "Maximum number of epochs.")
flags.DEFINE_float("dropout", 0.3, "Dropout rate.")
flags.DEFINE_integer(
    "document_len", 5120, 
    "Length of document in words the screenplay is split into.")
flags.DEFINE_integer(
    "overlap_len", 512, 
    "Number of words overlapping between successive documents of a screenplay.")
flags.DEFINE_integer(
    "subword_batch_size", 2, "Batch size of subword sequences for encoding.")
flags.DEFINE_integer(
    "cr_batch_size", 16, 
    "Batch size of word sequences for character head recognition.")
flags.DEFINE_integer(
    "fn_batch_size", 16, "Batch size of word pairs for fine scoring.")
flags.DEFINE_integer(
    "cr_seq_len", 256, 
    "Sequence length of word sequences for character head recognition.")

# Model
flags.DEFINE_integer(
    "topk", 50, 
    "Maximum number of preceding antecedents to retain after coarse scoring.")
flags.DEFINE_integer("gru_nlayers", 1, "Number of GRU layers.")
flags.DEFINE_integer("gru_hidden_size", 256, "Hidden size of GRU.")
flags.DEFINE_bool("gru_bi", True, "Bidirectional GRU layers.")
flags.DEFINE_integer("tag_embedding_size", 16, "Tag embedding size.")

# Validators
flags.register_validator(
    "document_len", lambda x: x in [2048, 3072, 4096, 5120], 
    "document_len should be 2048, 3072, 4096, or 5120")
flags.register_validator(
    "overlap_len", lambda x: x in [128, 256, 384, 512], 
    "overlap_len should be 128, 256, 384, or 512")

def main(argv):
    if len(argv) > 1:
        print(f"Unnecessary command-line arguments: {argv}")
    time = (datetime.datetime.now(pytz.timezone("America/Los_Angeles"))
            .strftime("%b%d_%I:%M:%S%p"))
    log_file = os.path.join(FLAGS.logs_dir, f"latest.log")
    train_file = os.path.join(
        FLAGS.input_dir, FLAGS.input_type, "train.jsonlines")
    dev_file = os.path.join(FLAGS.input_dir, FLAGS.input_type, "dev.jsonlines")
    trainer = CoreferenceTrainer(
        log_file=log_file,
        tag_embedding_size=FLAGS.tag_embedding_size,
        gru_nlayers=FLAGS.gru_nlayers,
        gru_hidden_size=FLAGS.gru_hidden_size,
        gru_bidirectional=FLAGS.gru_bi,
        topk=FLAGS.topk,
        dropout=FLAGS.dropout,
        weights_path=FLAGS.weights_file,
        train_path=train_file,
        dev_path=dev_file,
        freeze_bert=FLAGS.freeze_bert,
        genre=FLAGS.genre,
        bce_weight=FLAGS.bce_weight,
        bert_lr=FLAGS.bert_lr,
        lr=FLAGS.lr,
        weight_decay=FLAGS.weight_decay,
        max_epochs=FLAGS.max_epochs,
        document_len=FLAGS.document_len,
        overlap_len=FLAGS.overlap_len,
        cr_seq_len=FLAGS.cr_seq_len,
        subword_batch_size=FLAGS.subword_batch_size,
        cr_batch_size=FLAGS.cr_batch_size,
        fn_batch_size=FLAGS.fn_batch_size,
        save_model=FLAGS.save_model,
        save_output=FLAGS.save_tensors,
        save_loss_curve=FLAGS.save_loss_curves,
        debug=FLAGS.debug)
    trainer.run()

if __name__=="__main__":
    app.run(main)