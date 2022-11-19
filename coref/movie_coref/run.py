"""Main entry-point for coreference model training."""
# pyright: reportGeneralTypeIssues=false
from mica_text_coref.coref.movie_coref.coreference_trainer import CoreferenceTrainer

from absl import flags
from absl import app
import datetime
import os
import pytz

FLAGS = flags.FLAGS
proj_dir = os.path.join(os.getenv("PROJ_DIR"), "mica_text_coref")
data_dir = os.path.join(os.getenv("DATA_DIR"), "mica_text_coref")

# Directories and Files
flags.DEFINE_string("input_dir", default=os.path.join(data_dir, "movie_coref/results"), help="Directory of train and dev jsonlines.")
flags.DEFINE_enum("input_type", default="regular", enum_values=["regular", "nocharacters", "addsays"], help="Type of preprocessing applied to screenplays.")
flags.DEFINE_string("weights_file", default=os.path.join(data_dir, "word_level_coref/data/roberta_(e20_2021.05.02_01.16)_release.pt"),
    help="Filepath of word-level coreference roberta model's weights.")
flags.DEFINE_string("output_dir", default=os.path.join(data_dir, "movie_coref/results/coreference"), help="Directory to save model weights, predictions, evaluation metrics, and loss curves.")
flags.DEFINE_string("logs_dir", default=os.path.join(data_dir, "movie_coref/results/coreference/logs"), help="Directory to save logs.")
flags.DEFINE_string("reference_scorer", default=os.path.join(proj_dir, "coref/movie_coref/scorer/v8.01/scorer.pl"), help="Path to conll reference scorer.")

# Training
flags.DEFINE_bool("freeze_bert", default=False, help="Freeze RoBerta.")
flags.DEFINE_enum("genre", default="wb", enum_values=["bc", "bn", "mz", "nw", "pt", "tc", "wb"], help="Genre.")
flags.DEFINE_float("bce_weight", default=0.5, help="Weight of BCE coreference loss.")
flags.DEFINE_float("bert_lr", default=1e-5, help="Learning rate for the transformer.")
flags.DEFINE_float("character_lr", default=1e-5, help="Learning rate for the character recognition model.")
flags.DEFINE_float("coref_lr", default=1e-5, help="Learning rate for the coreference model.")
flags.DEFINE_float("weight_decay", default=1e-3, help="Weight decay.")
flags.DEFINE_integer("max_epochs", default=10, help="Maximum number of epochs.")
flags.DEFINE_float("dropout", default=0.3, help="Dropout rate.")
flags.DEFINE_integer("train_document_len", default=5120, help="Length of document in words the training screenplay is split into.")
flags.DEFINE_integer("eval_document_len", default=None, help="Length of document in words the evaluation screenplay is split into. None means the screenplay is not split")
flags.DEFINE_integer("eval_document_overlap_len", default=512, help="Number of words overlapping between successive documents of an evaluation screenplay.")
flags.DEFINE_integer("subword_batch_size", default=2, help="Batch size of subword sequences for encoding.")
flags.DEFINE_integer("cr_batch_size", default=16, help="Batch size of word sequences for character head recognition.")
flags.DEFINE_integer("fn_batch_size", default=16, help="Batch size of word pairs for fine scoring.")
flags.DEFINE_integer("sp_batch_size", default=4, help="Batch size of head ids for span prediction.")
flags.DEFINE_integer("cr_seq_len", default=256, help="Sequence length of word sequences for character head recognition.")
flags.DEFINE_bool("load_weights", default=True, help="Initialize roberta and coreference module with pretrained word-level coreference model's weights.")
flags.DEFINE_bool("run_span", default=False, help="Train and evaluate the span predictor module.")
flags.DEFINE_bool("add_cr_to_coarse", default=False, help="Add character scores to the coarse scores for top antecedent selection.")
flags.DEFINE_bool("filter_by_cr", default=False, help="Filter antecedents by the predicted character heads for finding predicted word-level clusters.")
flags.DEFINE_bool("remove_singleton_cr", default=False, help="Remove predicted word-level clusters containing a single predicted character head.")
flags.DEFINE_integer("train_cr_epochs", default=0, help="Number of initial epochs during which we only train the character recognition module.")
flags.DEFINE_bool("train_bert_with_cr", default=False, help="Train transformer along with the character recognition module in the initial train_cr_epochs")
flags.DEFINE_bool("eval_train", default=False, help="Evaluate training set.")
flags.DEFINE_integer("initial_no_eval_epochs", default=-1, help="Initial number of epochs for which no evaluation is performed")
flags.DEFINE_bool("save_model", default=False, help="Save model weights.")
flags.DEFINE_bool("save_tensors", default=False, help="Save predictions.")
flags.DEFINE_bool("save_loss_curves", default=False, help="Save loss curves.")
flags.DEFINE_bool("debug", default=False, help="Debug output.")
flags.DEFINE_bool("log_to_latest", default=False, help="Log to latest.log and don't create new log file.")

# Model
flags.DEFINE_integer("topk", default=50, help="Maximum number of preceding antecedents to retain after coarse scoring.")
flags.DEFINE_integer("gru_nlayers", default=1, help="Number of GRU layers.")
flags.DEFINE_integer("gru_hidden_size", default=256, help="Hidden size of GRU.")
flags.DEFINE_bool("gru_bi", default=False, help="Bidirectional GRU layers.")
flags.DEFINE_integer("tag_embedding_size", default=16, help="Tag embedding size.")

# Validators
flags.register_validator("train_document_len", lambda x: x in [2048, 3072, 4096, 5120], message="train_document_len should be one of 2048, 3072, 4096, or 5120")
flags.register_validator("eval_document_len", lambda x: x is None or x in [2048, 3072, 4096, 5120], message="eval_document_len should be None or one of 2048, 3072, 4096, or 5120")
flags.register_validator("eval_document_overlap_len", lambda x: x in [128, 256, 384, 512], message="eval_document_overlap_len should be one of 128, 256, 384, or 512")

def main(argv):
    if len(argv) > 1: print(f"Unnecessary command-line arguments: {argv}")
    if FLAGS.log_to_latest:
        log_file = os.path.join(FLAGS.logs_dir, f"latest.log")
    else:
        time = (datetime.datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%b%d_%I:%M:%S%p"))
        log_file = os.path.join(FLAGS.logs_dir, f"{time}.log")
    train_file = os.path.join(FLAGS.input_dir, FLAGS.input_type, "train.jsonlines")
    dev_file = os.path.join(FLAGS.input_dir, FLAGS.input_type, "dev.jsonlines")
    trainer = CoreferenceTrainer(
        log_file=log_file,
        output_dir=FLAGS.output_dir,
        reference_scorer=FLAGS.reference_scorer,
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
        character_lr=FLAGS.character_lr,
        coref_lr=FLAGS.coref_lr,
        weight_decay=FLAGS.weight_decay,
        max_epochs=FLAGS.max_epochs,
        train_document_len=FLAGS.train_document_len,
        eval_document_len=FLAGS.eval_document_len,
        eval_document_overlap_len=FLAGS.eval_document_overlap_len,
        subword_batch_size=FLAGS.subword_batch_size,
        cr_seq_len=FLAGS.cr_seq_len,
        cr_batch_size=FLAGS.cr_batch_size,
        fn_batch_size=FLAGS.fn_batch_size,
        sp_batch_size=FLAGS.sp_batch_size,
        load_pretrained_weights=FLAGS.load_weights,
        run_span=FLAGS.run_span,
        eval_train=FLAGS.eval_train,
        initial_epochs_no_eval=FLAGS.initial_no_eval_epochs,
        add_cr_to_coarse=FLAGS.add_cr_to_coarse,
        filter_mentions_by_cr=FLAGS.filter_by_cr,
        remove_singleton_cr=FLAGS.remove_singleton_cr,
        train_cr_epochs=FLAGS.train_cr_epochs,
        train_bert_with_cr=FLAGS.train_bert_with_cr,
        save_model=FLAGS.save_model,
        save_output=FLAGS.save_tensors,
        save_loss_curve=FLAGS.save_loss_curves,
        debug=FLAGS.debug)
    trainer()

if __name__=="__main__":
    app.run(main)