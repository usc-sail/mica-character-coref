"""Train a coreference longformer model on training dataset until performance
no longer improves on development dataset.
"""

from mica_text_coref.coref.seq_coref import coref_longformer
from mica_text_coref.coref.seq_coref import evaluation
from mica_text_coref.coref.seq_coref import utils
from mica_text_coref.coref.seq_coref import data_utils
from mica_text_coref.coref.seq_coref import inference

from absl import logging
import numpy as np
import os
import time
import torch
from torch.utils import data as tdata
from torch import optim
from transformers import get_linear_schedule_with_warmup

def train(tensors_dir:str,
          perl_scorer:str,
          output_dir:str,
          large_longformer=False,
          train_batch_size=64,
          infer_batch_size=64,
          learning_rate=1e-5,
          weight_decay=1e-3,
          print_n_batches=10,
          max_epochs=10,
          max_grad_norm=0.1,
          patience=3,
          warmup_ratio=0.1,
          evaluation_strategy="perl",
          evaluate_train=False,
          save_model=False,
          save_predictions=False
          ):
    """Train model on train dataset until performance no longer improves
    on dev dataset.
    """
    # Early stopping counters and other variables
    best_dev_F1 = -np.inf
    best_epoch = np.nan
    epochs_left = patience

    # Initialize model
    with utils.timer():
      logging.info("Initializing Coreference Longformer model...")
      model = coref_longformer.CorefLongformerModel(use_large=large_longformer)

    # Load train and dev tensors
    with utils.timer():
      logging.info(f"Loading train tensors...")
      train_dataset = data_utils.load_tensors(
        os.path.join(tensors_dir, "train"))
      logging.info(f"Loading dev tensors...")
      dev_dataset = data_utils.load_tensors(os.path.join(tensors_dir, "dev"))

    # Create dataloaders
    train_dataloader = tdata.DataLoader(
      train_dataset, batch_size=train_batch_size, shuffle=True)
    dev_dataloader = tdata.DataLoader(dev_dataset, batch_size=infer_batch_size)
    n_train_batches = len(train_dataloader)
    n_dev_batches = len(dev_dataloader)
    logging.info(f"Number of training batches = {n_train_batches}")
    logging.info(f"Number of inference batches = {n_dev_batches}")

    # Create optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                            weight_decay=weight_decay)
    n_training_steps = max_epochs * n_train_batches
    n_warmup_steps = int(warmup_ratio * n_training_steps)
    scheduler = get_linear_schedule_with_warmup(
      optimizer, num_warmup_steps=n_warmup_steps,
      num_training_steps=n_training_steps)
    logging.info(f"Number of warmup steps = {n_warmup_steps}")
    logging.info(f"Number of training steps = {n_training_steps}")
    
    # Training and evaluation loop
    with utils.timer():
        logging.info("Training started")
        for epoch in range(max_epochs):
            epoch_dir = os.path.join(output_dir, f"epoch_{epoch + 1}")
            epoch_train_dir = os.path.join(epoch_dir, "train")
            epoch_dev_dir = os.path.join(epoch_dir, "dev")
            os.makedirs(epoch_dir, exist_ok=True)
            os.makedirs(epoch_train_dir, exist_ok=True)
            os.makedirs(epoch_dev_dir, exist_ok=True)

            # Training for one epoch
            with utils.timer():
                logging.info(f"Epoch {epoch + 1} training started")
                model.train()
                running_batch_loss = []
                running_batch_train_time = []
                
                # Batch training loop
                for i, batch in enumerate(train_dataloader):
                    
                    # One training step
                    (batch_token_ids, batch_mention_ids, batch_label_ids,
                    batch_attn_mask, batch_global_attn_mask, _) = batch
                    batch_start_time = time.time()
                    optimizer.zero_grad()
                    batch_loss, _ = model(
                        batch_token_ids, batch_mention_ids, batch_attn_mask,
                        batch_global_attn_mask, batch_label_ids)
                    batch_loss.backward()
                    torch.nn.utils.clip_grad.clip_grad_norm_(
                    model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    batch_time_taken = time.time() - batch_start_time
                    running_batch_loss.append(batch_loss.detach().item())
                    running_batch_train_time.append(batch_time_taken)

                    # Log after print_n_batches
                    if (i + 1) % print_n_batches == 0:
                        average_batch_loss = np.mean(running_batch_loss)
                        average_batch_train_time = np.mean(
                            running_batch_train_time)
                        estimated_time_remaining = (
                            utils.convert_float_seconds_to_time_string(
                            average_batch_train_time * (n_train_batches-i-1)))
                        average_batch_train_time_str = (
                            utils.convert_float_seconds_to_time_string(
                            average_batch_train_time))
                        logging.info(f"Batch {i + 1}")
                        logging.info("Average training loss @ batch = "
                                     f"{average_batch_loss:.4f}")
                        logging.info("Average training time taken @ batch = "
                                     f"{average_batch_train_time_str}")
                        logging.info("Estimated training time remaining for "
                                     f"epoch {epoch + 1} = "
                                     f"{estimated_time_remaining}")
                        running_batch_loss = []
                        running_batch_train_time = []
                logging.info(f"Epoch {epoch + 1} training ended")            

            # Save model
            if save_model:
                logging.info(f"Saving model after epoch {epoch + 1}")
                utils.save_model(model, epoch_dir)

            # Inference and evaluation on training set
            if evaluate_train:
                with utils.timer():
                    logging.info(f"Epoch {epoch + 1} Inference & Evaluation on"
                                 " training set started")
                    (train_predictions, train_label_ids, train_attn_mask, 
                    train_doc_ids) = inference.infer(
                        train_dataloader, model, print_n_batches=print_n_batches)
                    train_metric = evaluation.evaluate(
                        train_label_ids, train_predictions, train_attn_mask,
                        train_doc_ids, evaluation_strategy)
                    logging.info(f"Training Performance = {train_metric.score}")
                    logging.info(f"Epoch {epoch + 1} Inference & Evaluation on"
                                 " training set ended")
                if save_predictions:
                    utils.save_predictions(
                        train_label_ids, train_predictions, train_doc_ids,
                        train_attn_mask, epoch_train_dir)
        
            # Inference and evaluation on dev set
            with utils.timer():
                logging.info(f"Epoch {epoch + 1} Inference & Evaluation on "
                              "dev set started")
                (dev_predictions, dev_label_ids, dev_attn_mask, 
                dev_doc_ids) = inference.infer(
                    dev_dataloader, model, print_n_batches=print_n_batches)
                dev_metric = evaluation.evaluate(
                    dev_label_ids, dev_predictions, dev_attn_mask,
                    dev_doc_ids, evaluation_strategy)
                logging.info(f"Dev Performance = {dev_metric.score}")
                logging.info(f"Epoch {epoch + 1} Inference & Evaluation on"
                             " dev set ended")
            if save_predictions:
                utils.save_predictions(
                    dev_label_ids, dev_predictions, dev_doc_ids,
                    dev_attn_mask, epoch_dev_dir)

            # Early-stopping
            logging.info("Checking for early-stopping")
            dev_F1 = dev_metric.score.f1
            if epoch == 0 or dev_F1 > best_dev_F1:
                epochs_left = patience
                best_epoch = epoch + 1
                if epoch > 0:
                    delta = 100 * (dev_F1 - best_dev_F1)
                    logging.info(f"Dev F1 improved by {delta:.1f}")
                best_dev_F1 = dev_F1
            else:
                epochs_left -= 1
                logging.info(f"Dev F1 is {-delta:.1f} lower than best Dev F1 "
                             f"({100*best_dev_F1:.1f})")
                logging.info(f"{epochs_left} epochs left until Dev F1 to"
                             " improve to avoid early-stopping!")
            if epochs_left == 0:
                logging.info("Early stopping!")
                break

            logging.info(f"Epoch {epoch + 1} done")
        logging.info("Training ended")

    logging.info(f"Best Dev F1 = {100*best_dev_F1:.1f}")
    logging.info(f"Best epoch = {best_epoch}")