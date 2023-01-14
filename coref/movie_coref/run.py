"""Main entry-point for coreference model training.

The movie coreference model can be trained in three different ways:

1. ALL          : Train on all full-length scripts with early-stopping on script excerpts.
2. LOMO         : Leave out one full-length script, train on the remaining full-length scripts
                  with early stopping on script excerpts, and then evaluate on the left out full length script.
3. LOMO_EXCERPTS: Leave out one full-length script, train on the remaining full-length scripts and script excerpts
                  with early stopping on the left out full-length script.

(LOMO is acronym for leave-one-movie-out)

By default, training is done according to ALL.
To do LOMO training, pass --test_movie=<movie_name>.
To do LOMO_EXCERPTS training, pass --test_movie=<movie_name> --train_excerpts.
<movie_name> has to be the name of one of the full-length scripts.
"""
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
flags.DEFINE_string("input_dir", default=os.path.join(data_dir, "movie_coref/results"),
                    help="Directory containing subdirectories of preprocessed script jsonlines.")
flags.DEFINE_enum("input_type", default="regular", enum_values=["regular", "nocharacters", "addsays"],
                    help="Type of preprocessing applied to screenplays.")
flags.DEFINE_string("weights_file",
                    default=os.path.join(data_dir, "word_level_coref/data/roberta_(e20_2021.05.02_01.16)_release.pt"), 
                    help="Filepath of word-level coreference roberta model's weights.")
flags.DEFINE_string("output_dir", default=os.path.join(data_dir, "movie_coref/results/coreference"),
                    help="Directory to save model weights, predictions, metrics, loss curves, and logs.")
flags.DEFINE_string("reference_scorer", default=os.path.join(proj_dir, "coref/movie_coref/scorer/v8.01/scorer.pl"),
                    help="Path of conll reference scorer.")

# Training Mode
flags.DEFINE_bool("train_excerpts", default=False, help="Train excerpts.")
flags.DEFINE_enum("test_movie", default="none",
                  enum_values=["avengers_endgame", "dead_poets_society", "john_wick", "prestige", "quiet_place",
                               "zootopia", "none"],
                  help="Left out movie used for testing.")
flags.DEFINE_bool("hierarchical", default=False, help="Run hierarchical inference.")

# Hyperparameters
flags.DEFINE_integer("topk", default=50, help="Maximum number of preceding antecedents to retain after coarse scoring.")
flags.DEFINE_integer("repk", default=3, help="Number of representative mentions to sample per cluster.")
flags.DEFINE_bool("load_bert", default=True, help="Load transformer weights from word-level coreference model, "
                                                  "otherwise regular roberta-large weights are used.")
flags.DEFINE_bool("freeze_bert", default=False, help="Freeze transformer.")
flags.DEFINE_enum("genre", default="wb", enum_values=["bc", "bn", "mz", "nw", "pt", "tc", "wb"], help="Genre.")
flags.DEFINE_float("bce_weight", default=0.5, help="Weight of the BCE coreference loss.")
flags.DEFINE_float("bert_lr", default=2e-5, help="Learning rate of the transformer.")
flags.DEFINE_float("character_lr", default=2e-4, help="Learning rate of the character recognition model.")
flags.DEFINE_float("coref_lr", default=2e-4, help="Learning rate of the coreference models.")
flags.DEFINE_float("warmup_steps", default=-1,
                   help=("Number of training steps when learning rate increases from 0 to max linearly."
                         " If -1, learning rate is constant."))
flags.DEFINE_float("weight_decay", default=1e-3, help="Weight decay.")
flags.DEFINE_integer("max_epochs", default=20, help="Maximum number of epochs for which to train the model.")
flags.DEFINE_integer("patience", default=5, help="Maximum number of epochs to wait for development set's performance "
                                                 "to improve until early-stopping.")
flags.DEFINE_float("dropout", default=0, help="Dropout rate.")
flags.DEFINE_integer("train_document_len", default=5120, help="Length of training subdocument in words.")
flags.DEFINE_integer("train_overlap_len", default=0,
                     help="Length of overlap between successive training subdocuments in words.")
flags.DEFINE_integer("dev_document_len", default=5120, help="Length of development subdocument in words.")
flags.DEFINE_integer("dev_overlap_len", default=512,
                     help="Length of overlap between successive development subdocuments in words.")
flags.DEFINE_enum("dev_merge_strategy", default="avg", enum_values=["none", "pre", "post", "max", "min", "avg"],
                  help="Type of merge to perform on coreference scores of adjacent development subdocuments.")
flags.DEFINE_integer("subword_batch_size", default=64, help="Batch size of subword sequences.")
flags.DEFINE_integer("cr_batch_size", default=64, help="Batch size of word sequences for character head recognition.")
flags.DEFINE_integer("fn_batch_size", default=64, help="Batch size of word pairs for fine scoring.")
flags.DEFINE_integer("sp_batch_size", default=64, help="Batch size of head ids for Span prediction.")
flags.DEFINE_integer("cr_seq_len", default=256,
                     help="Sequence length of word sequences for character head recognition.")
flags.DEFINE_bool("add_cr_to_coarse", default=True,
                  help="Add character scores to the coarse scores for top antecedent selection.")
flags.DEFINE_bool("filter_by_cr", default=False, help="Filter antecedents by the predicted character heads "
                                                      "for finding predicted word-level clusters.")
flags.DEFINE_bool("remove_singleton_cr", default=True,
                  help="Remove predicted word-level clusters containing a single predicted character head.")

# Model
flags.DEFINE_integer("gru_nlayers", default=1, help="Number of GRU layers.")
flags.DEFINE_integer("gru_hidden_size", default=256, help="Hidden size of GRU.")
flags.DEFINE_bool("gru_bi", default=True, help="Bidirectional GRU layers.")
flags.DEFINE_integer("tag_embedding_size", default=16, help="Tag embedding size.")

# Test settings
flags.DEFINE_multi_integer("test_document_lens", default=[2048, 3072, 4096, 5120],
                           help="Length of test subdocument in words.")
flags.DEFINE_multi_integer("test_overlap_lens", default=[0, 128, 256, 384, 512],
                           help="Length of overlap between successive test subdocuments in words.")
flags.DEFINE_multi_enum("test_merge_strategies", default=["none", "pre", "post", "max", "min", "avg"],
                        enum_values=["none", "pre", "post", "max", "min", "avg"],
                        help="Types of merge to perform on scores of adjacent test subdocuments.")

# Evaluation settings
flags.DEFINE_bool("eval_train", default=True, help="Evaluate training set.")
flags.DEFINE_integer("n_epochs_no_eval", default=0,
                     help="Initial number of epochs for which no evaluation is performed.")
flags.DEFINE_bool("save_log", default=False, help="Save log.")
flags.DEFINE_bool("save_model", default=False, help="Save model weights.")
flags.DEFINE_bool("save_predictions", default=False, help="Save predictions.")
flags.DEFINE_bool("save_loss_curve", default=False, help="Save loss curves.")
flags.DEFINE_bool("save", default=False, help="shortcut for --save_log --save_model --save_predictions "
                                              "--save_loss_curve.")
flags.DEFINE_bool("debug", default=False, help="Debug mode. Logs, models, predictions, and loss curves are not saved.")

# Validators
flags.register_multi_flags_validator(["train_excerpts", "test_movie"],
                                     lambda x: not x["train_excerpts"] or x["test_movie"] != "none",
                                     message="test_movie can not be 'none' if --train_excerpts is passed")
flags.register_multi_flags_validator(["save", "debug"], lambda x: not(x["save"] and x["debug"]),
                                     message="debug and save cannot both be true.")

def main(argv):
    # Return if extra command-line arguments are provided
    if len(argv) > 1:
        print(f"Extra command-line arguments: {argv}")
        return
    
    # Create output directory name from current system time
    # Append flags to indicate the training mode
    time = datetime.datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%b%d_%I:%M:%S%p")
    output_dir = os.path.join(FLAGS.output_dir, time)
    if FLAGS.test_movie != "none":
        output_dir += "_" + FLAGS.test_movie
    if FLAGS.train_excerpts:
        output_dir += "_excerpts"
    if FLAGS.hierarchical:
        output_dir += "_hi"
    output_dir += f"_{os.getpid()}"

    # Get the full-length and excerpts screenplays
    full_lengths_file = os.path.join(FLAGS.input_dir, FLAGS.input_type, "train.jsonlines")
    excerpts_file = os.path.join(FLAGS.input_dir, FLAGS.input_type, "dev.jsonlines")

    # Initialize the trainer
    trainer = CoreferenceTrainer(
        preprocess=FLAGS.input_type,
        output_dir=output_dir,
        reference_scorer_file=FLAGS.reference_scorer,
        full_length_scripts_file=full_lengths_file,
        excerpts_file=excerpts_file,
        weights_file=FLAGS.weights_file,
        train_excerpts=FLAGS.train_excerpts,
        test_movie=FLAGS.test_movie,
        hierarchical=FLAGS.hierarchical,
        tag_embedding_size=FLAGS.tag_embedding_size,
        gru_nlayers=FLAGS.gru_nlayers,
        gru_hidden_size=FLAGS.gru_hidden_size,
        gru_bidirectional=FLAGS.gru_bi,
        topk=FLAGS.topk,
        n_representative_mentions=FLAGS.repk,
        dropout=FLAGS.dropout,
        freeze_bert=FLAGS.freeze_bert,
        load_bert=FLAGS.load_bert,
        genre=FLAGS.genre,
        bce_weight=FLAGS.bce_weight,
        bert_lr=FLAGS.bert_lr,
        character_lr=FLAGS.character_lr,
        coref_lr=FLAGS.coref_lr,
        warmup_steps=FLAGS.warmup_steps,
        weight_decay=FLAGS.weight_decay,
        max_epochs=FLAGS.max_epochs,
        patience=FLAGS.patience,
        train_document_len=FLAGS.train_document_len,
        train_overlap_len=FLAGS.train_overlap_len,
        dev_document_len=FLAGS.dev_document_len,
        dev_overlap_len=FLAGS.dev_overlap_len,
        dev_merge_strategy=FLAGS.dev_merge_strategy,
        test_document_lens=FLAGS.test_document_lens,
        test_overlap_lens=FLAGS.test_overlap_lens,
        test_merge_strategies=FLAGS.test_merge_strategies,
        subword_batch_size=FLAGS.subword_batch_size,
        cr_seq_len=FLAGS.cr_seq_len,
        cr_batch_size=FLAGS.cr_batch_size,
        fn_batch_size=FLAGS.fn_batch_size,
        sp_batch_size=FLAGS.sp_batch_size,
        evaluate_train=FLAGS.eval_train,
        n_epochs_no_eval=FLAGS.n_epochs_no_eval,
        add_cr_to_coarse=FLAGS.add_cr_to_coarse,
        filter_mentions_by_cr=FLAGS.filter_by_cr,
        remove_singleton_cr=FLAGS.remove_singleton_cr,
        save_log=FLAGS.save_log or FLAGS.save and not FLAGS.debug,
        save_model=FLAGS.save_model or FLAGS.save and not FLAGS.debug,
        save_predictions=FLAGS.save_predictions or FLAGS.save and not FLAGS.debug,
        save_loss_curve=FLAGS.save_loss_curve or FLAGS.save and not FLAGS.debug
        )
    
    # Run trainer
    trainer()

if __name__=="__main__":
    app.run(main)