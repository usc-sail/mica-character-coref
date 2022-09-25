"""Train a coreference longformer model on training dataset until performance
no longer improves on development dataset.
"""

from mica_text_coref.coref.seq_coref import acceleration
from mica_text_coref.coref.seq_coref import coref_longformer
from mica_text_coref.coref.seq_coref import evaluation
from mica_text_coref.coref.seq_coref import utils
from mica_text_coref.coref.seq_coref import data_utils
from mica_text_coref.coref.seq_coref import inference

import numpy as np
import os
import time
from torch.utils import data as tdata
from torch import optim
from transformers import get_linear_schedule_with_warmup, Adafactor

def train(tensors_dir:str,
          perl_scorer:str,
          output_dir:str,
          large_longformer=False,
          train_batch_size=16,
          infer_batch_size=16,
          learning_rate=1e-5,
          weight_decay=1e-3,
          use_adafactor=False,
          grad_checkpointing=False,
          use_scheduler=False,
          warmup_ratio=0.1,
          max_epochs=10,
          max_grad_norm=0.1,
          patience=3,
          print_n_batches=10,
          evaluation_strategy="perl",
          evaluate_train=False,
          save_model=False,
          save_predictions=False
          ):
    """Train model on train dataset until performance no longer improves
    on dev dataset.
    """
    # Get accelerator and logger
    accelerator, logger = acceleration.accelerator, acceleration.logger

    # Early stopping counters and other variables
    best_dev_F1 = -np.inf
    best_epoch = np.nan
    epochs_left = patience

    # Initialize model
    with utils.timer("model initialization"):
      model = coref_longformer.CorefLongformerModel(
        use_large=large_longformer, grad_checkpointing=grad_checkpointing)

    # Load train and dev tensors
    with utils.timer("data loading"):
      train_dataset = data_utils.load_tensors(os.path.join(tensors_dir,"train"))
      dev_dataset = data_utils.load_tensors(os.path.join(tensors_dir,"dev"))
      train_size = len(train_dataset)
      dev_size = len(dev_dataset)
    
    # Create dataloaders, optimizer, and scheduler
    train_dataloader = tdata.DataLoader(
      train_dataset, batch_size=train_batch_size, shuffle=True)
    dev_dataloader = tdata.DataLoader(dev_dataset, batch_size=infer_batch_size)
    if use_adafactor:
        optimizer = Adafactor(model.parameters(), lr=learning_rate,
                              weight_decay=weight_decay, scale_parameter=False,
                              relative_step=False, warmup_init=False)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                weight_decay=weight_decay)
    
    # Accelerate model, dataloaders, and optimizer
    model, train_dataloader, dev_dataloader, optimizer = accelerator.prepare(
        model, train_dataloader, dev_dataloader, optimizer)

    # Log number of training and inference batches, and number of training steps
    n_train_batches = len(train_dataloader)
    n_dev_batches = len(dev_dataloader)
    effective_train_batch_size = round(train_size/n_train_batches)
    effective_dev_batch_size = round(dev_size/n_dev_batches)
    n_training_steps = max_epochs * n_train_batches
    logger.info(f"Effective train batch size = {effective_train_batch_size}")
    logger.info(f"Effective dev batch size = {effective_dev_batch_size}")
    logger.info(f"Number of training batches = {n_train_batches}")
    logger.info(f"Number of inference batches = {n_dev_batches}")
    logger.info(f"Number of training steps = {n_training_steps}")

    # Initialize and accelerate scheduler
    if use_scheduler:
        n_warmup_steps = int(warmup_ratio * n_training_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=n_warmup_steps,
            num_training_steps=n_training_steps)
        scheduler = accelerator.prepare_scheduler(scheduler)
        logger.info(f"Number of warmup steps = {n_warmup_steps}")
    
    # Training and evaluation loop
    with utils.timer("training"):
        for epoch in range(max_epochs):
            
            # Create epoch directories
            epoch_dir = os.path.join(output_dir, f"epoch_{epoch + 1}")
            epoch_train_dir = os.path.join(epoch_dir, "train")
            epoch_dev_dir = os.path.join(epoch_dir, "dev")
            os.makedirs(epoch_dir, exist_ok=True)
            os.makedirs(epoch_train_dir, exist_ok=True)
            os.makedirs(epoch_dev_dir, exist_ok=True)

            # Training for one epoch
            with utils.timer(f"epoch {epoch + 1} training"):
                model.train()
                running_batch_loss = []
                running_batch_train_time = []
                
                # Batch training loop
                for i, batch in enumerate(train_dataloader):
                    batch_start_time = time.time()
                    
                    # One training step
                    with accelerator.accumulate(model):
                        optimizer.zero_grad()
                        with accelerator.autocast():
                            (batch_token_ids, batch_mention_ids,
                            batch_label_ids, batch_attn_mask,
                            batch_global_attn_mask, _) = batch
                            batch_loss = model(
                                batch_token_ids, batch_mention_ids,
                                batch_attn_mask, batch_global_attn_mask,
                                batch_label_ids)
                        accelerator.backward(batch_loss)
                        if optimizer.gradient_state.sync_gradients:
                            accelerator.clip_grad_norm_(model.parameters(),
                                                        max_grad_norm)
                        optimizer.step()
                        if use_scheduler and (
                           not accelerator.optimizer_step_was_skipped):
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
                        logger.info(f"Batch {i + 1}")
                        logger.info("Average training loss @ batch = "
                                     f"{average_batch_loss:.4f}")
                        logger.info("Average training time taken @ batch = "
                                     f"{average_batch_train_time_str}")
                        logger.info("Estimated training time remaining for "
                                     f"epoch {epoch + 1} = "
                                     f"{estimated_time_remaining}")
                        running_batch_loss = []
                        running_batch_train_time = []

            # Wait for all process to complete
            accelerator.wait_for_everyone()

            # Save model
            if save_model:
                logger.info(f"Saving model after epoch {epoch + 1}")
                unwrapped_model = accelerator.unwrap_model(model)
                utils.save_model(unwrapped_model, epoch_dir)

            # Inference and evaluation on training set
            if evaluate_train:
                with utils.timer(f"epoch {epoch + 1} training inference and evaluation"):
                    (train_predictions, train_label_ids, train_attn_mask, 
                    train_doc_ids) = inference.infer(
                        train_dataloader, model,
                        print_n_batches=print_n_batches)
                    train_metric = evaluation.evaluate(
                        train_label_ids, train_predictions, train_attn_mask,
                        train_doc_ids, perl_scorer, evaluation_strategy)
                    logger.info(f"Training Performance = {train_metric.score}")
                accelerator.wait_for_everyone()
                if save_predictions:
                    logger.info("Saving train predictions after epoch"
                                f" {epoch + 1}")
                    utils.save_predictions(
                        train_label_ids, train_predictions, train_doc_ids,
                        train_attn_mask, epoch_train_dir)
        
            # Inference and evaluation on dev set
            with utils.timer(f"epoch {epoch + 1} dev inference and evaluation"):
                (dev_predictions, dev_label_ids, dev_attn_mask, 
                dev_doc_ids) = inference.infer(
                    dev_dataloader, model, print_n_batches=print_n_batches)
                dev_metric = evaluation.evaluate(
                    dev_label_ids, dev_predictions, dev_attn_mask,
                    dev_doc_ids, perl_scorer, evaluation_strategy)
                logger.info(f"Dev Performance = {dev_metric.score}")
            accelerator.wait_for_everyone()
            if save_predictions:
                logger.info(f"Saving dev predictions after epoch {epoch + 1}")
                utils.save_predictions(
                    dev_label_ids, dev_predictions, dev_doc_ids,
                    dev_attn_mask, epoch_dev_dir)

            # Early-stopping
            logger.info("Checking for early-stopping")
            dev_F1 = dev_metric.score.f1
            if epoch == 0 or dev_F1 > best_dev_F1:
                epochs_left = patience
                best_epoch = epoch + 1
                if epoch > 0:
                    delta = 100 * (dev_F1 - best_dev_F1)
                    logger.info(f"Dev F1 improved by {delta:.1f}")
                best_dev_F1 = dev_F1
            else:
                epochs_left -= 1
                logger.info(f"Dev F1 is {-delta:.1f} lower than best Dev F1 "
                             f"({100*best_dev_F1:.1f})")
                logger.info(f"{epochs_left} epochs left until Dev F1 to"
                             " improve to avoid early-stopping!")
            if epochs_left == 0:
                logger.info("Early stopping!")
                break

            logger.info(f"Epoch {epoch + 1} done")

    logger.info(f"Best Dev F1 = {100*best_dev_F1:.1f}")
    logger.info(f"Best epoch = {best_epoch}")