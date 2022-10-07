"""Main entry-point for training character recognition model.
"""
from mica_text_coref.coref.movie_coref import character_recognition
from mica_text_coref.coref.movie_coref import data
from mica_text_coref.coref.movie_coref import models

from absl import app
from absl import flags
import accelerate
from accelerate import logging as alogging
import datetime
import logging
import os
import pytz
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer

proj_dir = os.getcwd()
FLAGS = flags.FLAGS

# Saving outputs
flags.DEFINE_string("output_dir",
                    default=os.path.join(proj_dir, "data/movie_coref/results/"
                                                   "character_recognition"),
                    help="Directory to which the model and inference output "
                         "will be saved.")
flags.DEFINE_bool("save_model", default=False,
                  help="If true, model weights are saved after every epoch.")
flags.DEFINE_bool("save_tensors", default=False,
                  help="If true, tensors of the dev inference output are "
                       "saved after every epoch.")
flags.DEFINE_list("save_tensors_names", default=["labels","logits"],
                  help="If save_tensors is true, the names of the tensors "
                       "to save.")

# Dataset creation
flags.DEFINE_string("input_dir",
                    default=os.path.join(proj_dir, "data/movie_coref/results"),
                    help="Directory containing the movie jsonlines "
                         "directories.")
flags.DEFINE_enum("format_type", default="regular",
                  enum_values=["regular","nocharacters","addsays"],
                  help="Format type of the input movie jsonlines.")
flags.DEFINE_string("encoder", default="roberta-base",
                    help="Huggingface model name of the encoder to use.")
flags.DEFINE_integer("seqlen", default=256, lower_bound=16,
                     help="Number of tokens to include in a training sequence.")
flags.DEFINE_bool("obey_scene_boundaries", default=False,
                  help="If true, training sequences do not cross scene "
                       "boundaries.")
flags.DEFINE_enum("label_type", default="head", enum_values=["head","span"],
                  help="Label type. If 'head', then only head word labels are "
                       "used. If 'span', full mention labels are used.")

# Model
flags.DEFINE_integer("parse_embedding_size", default=32,
                     help="Embedding size of the screenplay parse tags.")
flags.DEFINE_integer("gru_hidden_size", default=256,
                     help="Hidden size of the GRU.")
flags.DEFINE_integer("gru_num_layers", default=1,
                     help="Number of GRU layers.")
flags.DEFINE_float("gru_dropout", default=0, help="Dropout probability of GRU.")
flags.DEFINE_bool("gru_bidirectional", default=True,
                  help="If true, use bidirectional layers in GPU.")

# Training and Inference
flags.DEFINE_integer("train_batch_size", default=16,
                     help="Per-GPU training batch size.")
flags.DEFINE_integer("infer_batch_size", default=16,
                     help="Per-GPU inference batch size.")
flags.DEFINE_float("lr", default=2e-5, help="learning rate.")
flags.DEFINE_float("weight_decay", default=1e-3,
                   help="L2 regularization loss.")
flags.DEFINE_list("class_weights", default=None,
                  help="Custom class weights to use in cross entropy loss. "
                       "Should be a list of numbers with length equal to the "
                       "number of labels.")
flags.DEFINE_bool("use_scheduler", default=False,
                  help="If true, use linear scheduler for learning rate and "
                       "either warmup_ratio and warmup_steps should be set.")
flags.DEFINE_float("warmup_ratio", default=None,
                   help="If use_scheduler is true, warmup_ratio is the "
                        "fraction of total training steps to be used in the "
                        "linear warmup. If warmup_steps is provided, "
                        "warmup_ratio is ignored.")
flags.DEFINE_integer("warmup_steps", default=None,
                     help="If use scheduler is true, warmup_steps is the "
                          "number of training steps to be used in the linear "
                          "warmup. This option supercedes the warmup_ratio "
                          "option.")
flags.DEFINE_integer("max_epochs", default=10,
                     help="Maximum number of epochs to train for.")
flags.DEFINE_integer("patience", default=2,
                     help="Number of epochs to wait for the dev set "
                          "performance to improve before early-stopping.")
flags.DEFINE_float("max_grad_norm", default=None,
                   help="Cutoff gradient norm to be used in gradient "
                        "clipping. If none, gradient clipping is not done.")
flags.DEFINE_bool("evaluate_train", default=False,
                  help="If true, evaluate on training set as well.")

# Accelerator arguments
flags.DEFINE_bool("mixed_precision", default=False,
                  help="If true, use mixed-precision in training and "
                       "inference.")
flags.DEFINE_bool("gradient_checkpointing", default=False,
                  help="If true, use gradient checkpointing in training.")
flags.DEFINE_integer("gradient_accumulation", default=1,
                     help="The number of batches over which to accumulate "
                          "gradients. If none, gradient accumulation is not "
                          "performed.")

# Logging
flags.DEFINE_string("logs_dir", default=os.path.join(proj_dir, "data/logs"),
                    help="Logs directory. A new log file with current "
                         "timestamp will be created.")
flags.DEFINE_integer("log_frequency", default=10,
                     help="Number of batches after which to log training loss.")

def main(argv):
    accelerator = accelerate.Accelerator(
        mixed_precision="fp16" if FLAGS.mixed_precision else "no",
        gradient_accumulation_steps=FLAGS.gradient_accumulation)
    logger = alogging.get_logger("")

    # Add file handler
    if accelerator.is_local_main_process:
        time = (datetime.datetime.now(pytz.timezone("America/Los_Angeles"))
                    .strftime("%b%d_%I:%M:%S%p"))
        file_handler = logging.FileHandler(
            os.path.join(FLAGS.logs_dir, f"cr_{time}.log"))
        logger.logger.addHandler(file_handler)

    # Log command-line arguments
    for module_name, flag_items in FLAGS.flags_by_module_dict().items():
        if os.path.join(os.getcwd(), module_name) == __file__:
            for flag_item in flag_items:
                logger.info(f"FLAGS: {flag_item.name:<25s} = "
                            f"{flag_item._value}")
    
    # Character Recognition datasets, models, trainers
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.encoder, use_fast=True)
    dataloaders = []

    for partition in ["train", "dev"]:
        corpus = data.CorefCorpus(
            os.path.join(FLAGS.input_dir, FLAGS.format_type,
                         f"{partition}.jsonlines"))
        dataset = data.CharacterRecognitionDataset(
            corpus, tokenizer, seq_length=FLAGS.seqlen,
            obey_scene_boundaries=FLAGS.obey_scene_boundaries,
            label_type=FLAGS.label_type)
        if partition == "train":
            dataloader = DataLoader(dataset, batch_size=FLAGS.train_batch_size,
                                    shuffle=True)
        else:
            dataloader = DataLoader(dataset, batch_size=FLAGS.infer_batch_size)
        dataloaders.append(dataloader)
    
    train_dataloader, dev_dataloader = dataloaders
    num_parse_tags = len(set(train_dataloader.dataset.parse_tag_to_id.values()))
    num_labels = train_dataloader.dataset.num_labels
    class_weights = [float(wt) for wt in FLAGS.class_weights] if (
        FLAGS.class_weights) is not None else None
    model = models.CharacterRecognition(
        FLAGS.encoder, num_parse_tags=num_parse_tags,
        parse_tag_embedding_size=FLAGS.parse_embedding_size,
        gru_hidden_size=FLAGS.gru_hidden_size,
        gru_num_layers=FLAGS.gru_num_layers, gru_dropout=FLAGS.gru_dropout,
        gru_bidirectional=FLAGS.gru_bidirectional,
        num_labels=num_labels,
        gradient_checkpointing=FLAGS.gradient_checkpointing,
        class_weights=class_weights)
    optimizer = AdamW(model.parameters(), lr=FLAGS.lr,
                      weight_decay=FLAGS.weight_decay)
    trainer = character_recognition.CharacterRecognitionTrainer(
        accelerator=accelerator, 
        logger=logger, 
        model=model, 
        train_dataloader=train_dataloader, 
        dev_dataloader=dev_dataloader, 
        optimizer=optimizer, 
        use_scheduler=FLAGS.use_scheduler, 
        warmup_ratio=FLAGS.warmup_ratio, 
        warmup_steps=FLAGS.warmup_steps, 
        max_epochs=FLAGS.max_epochs, 
        max_grad_norm=FLAGS.max_grad_norm, 
        patience=FLAGS.patience, 
        log_batch_frequency=FLAGS.log_frequency, 
        evaluate_train=FLAGS.evaluate_train, 
        save_model=FLAGS.save_model, 
        save_tensors=FLAGS.save_tensors, 
        save_tensors_names=FLAGS.save_tensors_names, 
        save_dir=FLAGS.output_dir)
    
    # Run trainer
    trainer.run()

if __name__ == '__main__':
    app.run(main)