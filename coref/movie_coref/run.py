"""Main entry-point for coreference model training."""
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
flags.DEFINE_string("input_dir", default=os.path.join(data_dir, "movie_coref/results"), help="directory containing subdirectories of preprocessed train and dev script jsonlines")
flags.DEFINE_enum("input_type", default="nocharacters", enum_values=["regular", "nocharacters", "addsays"], help="type of preprocessing applied to screenplays")
flags.DEFINE_string("weights_file", default=os.path.join(data_dir, "word_level_coref/data/roberta_(e20_2021.05.02_01.16)_release.pt"), 
                    help="filepath of word-level coreference roberta model's weights")
flags.DEFINE_string("output_dir", default=os.path.join(data_dir, "movie_coref/results/coreference"), help="directory to save model weights, predictions, metrics, loss curves, and logs")
flags.DEFINE_string("reference_scorer", default=os.path.join(proj_dir, "coref/movie_coref/scorer/v8.01/scorer.pl"), help="path of conll reference scorer")

# Training
flags.DEFINE_bool("freeze_bert", default=False, help="freeze RoBerta transformer")
flags.DEFINE_enum("genre", default="wb", enum_values=["bc", "bn", "mz", "nw", "pt", "tc", "wb"], help="genre")
flags.DEFINE_float("bce_weight", default=0.5, help="weight of the BCE coreference loss")
flags.DEFINE_float("bert_lr", default=1e-5, help="learning rate of the transformer")
flags.DEFINE_float("character_lr", default=1e-4, help="learning rate of the character recognition model")
flags.DEFINE_float("coref_lr", default=1e-4, help="learning rate of the coreference model")
flags.DEFINE_float("warmup", default=-1, help="number of warmup epochs when the learning rate increases from 0 to max, can be a fraction. if -1, learning rate is constant")
flags.DEFINE_float("weight_decay", default=0, help="weight decay")
flags.DEFINE_integer("max_epochs", default=20, help="maximum number of epochs to train the model")
flags.DEFINE_integer("patience", default=3, help="maximum number of epochs to wait for dev performance to improve until early-stopping")
flags.DEFINE_float("dropout", default=0, help="dropout rate")
flags.DEFINE_integer("train_document_len", default=5120, help="length of training subdocument in words")
flags.DEFINE_integer("subword_batch_size", default=64, help="batch size of subword sequences")
flags.DEFINE_integer("cr_batch_size", default=64, help="batch size of word sequences for character head recognition")
flags.DEFINE_integer("fn_batch_size", default=64, help="batch size of word pairs for fine scoring")
flags.DEFINE_integer("sp_batch_size", default=64, help="batch size of head ids for span prediction")
flags.DEFINE_integer("cr_seq_len", default=256, help="sequence length of word sequences for character head recognition")
flags.DEFINE_bool("add_cr_to_coarse", default=True, help="if true, add character scores to the coarse scores for top antecedent selection")
flags.DEFINE_bool("filter_by_cr", default=False, help="if true, filter antecedents by the predicted character heads for finding predicted word-level clusters")
flags.DEFINE_bool("remove_singleton_cr", default=True, help="if true, remove predicted word-level clusters containing a single predicted character head")
flags.DEFINE_bool("eval_train", default=True, help="if true, evaluate training set")
flags.DEFINE_integer("n_epochs_no_eval", default=0, help="initial number of epochs for which no evaluation is performed")
flags.DEFINE_bool("save_log", default=False, help="if true, save log")
flags.DEFINE_bool("save_model", default=False, help="if true, save model weights")
flags.DEFINE_bool("save_predictions", default=False, help="if true, save predictions")
flags.DEFINE_bool("save_loss_curve", default=False, help="if true, save loss curves")
flags.DEFINE_bool("save", default=False, help="shortcut for save_log/model/predictions/loss_curve")
flags.DEFINE_bool("debug", default=False, help="debug mode, no logs, models or predictions are saved")

# Model
flags.DEFINE_integer("topk", default=50, help="maximum number of preceding antecedents to retain after coarse scoring")
flags.DEFINE_integer("gru_nlayers", default=1, help="number of GRU layers")
flags.DEFINE_integer("gru_hidden_size", default=256, help="hidden size of GRU")
flags.DEFINE_bool("gru_bi", default=True, help="bidirectional GRU layers")
flags.DEFINE_integer("tag_embedding_size", default=16, help="tag embedding size")

# Validators
flags.register_validator("train_document_len", lambda x: x in [1024, 2048, 3072, 4096, 5120], message="train_document_len should be one of 1024, 2048, 3072, 4096, or 5120")
flags.register_multi_flags_validator(["save", "debug"], lambda x: not(x["save"] and x["debug"]), "debug and save cannot both be true")

def main(argv):
    if len(argv) > 1:
        print(f"Extra command-line arguments: {argv}")
        return
    time = datetime.datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%b%d_%I:%M:%S%p")
    output_dir = os.path.join(FLAGS.output_dir, time)
    train_file = os.path.join(FLAGS.input_dir, FLAGS.input_type, "train.jsonlines")
    dev_file = os.path.join(FLAGS.input_dir, FLAGS.input_type, "dev.jsonlines")
    trainer = CoreferenceTrainer(
        preprocess=FLAGS.input_type,
        output_dir=output_dir,
        reference_scorer_file=FLAGS.reference_scorer,
        train_file=train_file,
        dev_file=dev_file,
        weights_file=FLAGS.weights_file,
        tag_embedding_size=FLAGS.tag_embedding_size,
        gru_nlayers=FLAGS.gru_nlayers,
        gru_hidden_size=FLAGS.gru_hidden_size,
        gru_bidirectional=FLAGS.gru_bi,
        topk=FLAGS.topk,
        dropout=FLAGS.dropout,
        freeze_bert=FLAGS.freeze_bert,
        genre=FLAGS.genre,
        bce_weight=FLAGS.bce_weight,
        bert_lr=FLAGS.bert_lr,
        character_lr=FLAGS.character_lr,
        coref_lr=FLAGS.coref_lr,
        warmup_epochs=FLAGS.warmup,
        weight_decay=FLAGS.weight_decay,
        max_epochs=FLAGS.max_epochs,
        patience=FLAGS.patience,
        train_document_len=FLAGS.train_document_len,
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
        save_loss_curve=FLAGS.save_loss_curve or FLAGS.save and not FLAGS.debug)
    trainer()

if __name__=="__main__":
    app.run(main)