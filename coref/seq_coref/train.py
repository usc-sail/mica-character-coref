"""Train a coreference longformer model on training dataset until performance
no longer improves on development dataset.
"""

from mica_text_coref.coref.seq_coref import coref_longformer
from mica_text_coref.coref.seq_coref import data
from mica_text_coref.coref.seq_coref import evaluate
from mica_text_coref.coref.seq_coref import util

from absl import logging
import getpass
import numpy as np
import os
import time
import torch
from torch.utils import data as tdata
from torch import nn
from torch import optim
from transformers import get_linear_schedule_with_warmup

def train(model: coref_longformer.CorefLongformerModel,
          train_dataset: tdata.TensorDataset,
          dev_dataset: tdata.TensorDataset,
          train_corpus: data.CorefCorpus | None = None,
          dev_corpus: data.CorefCorpus | None = None,
          use_gpu = False,
          use_data_parallel = False,
          train_batch_size = 64,
          eval_batch_size = 64,
          learning_rate = 1e-5,
          weight_decay = 1e-3,
          print_n_batches = 10,
          max_epochs = 10,
          max_grad_norm = 0.1,
          patience = 3,
          n_warmup_steps = 50,
          use_official_evaluation = False,
          official_scorer: str | None = None,
          evaluate_on_train = False,
          use_dynamic_loading = False,
          evaluate_against_original = False,
          save_model = False,
          save_model_after_every_epoch = False,
          save_predictions = False,
          save_predictions_after_every_epoch = False,
          save_directory: str | None = None,
          evaluate_seq = False
          ) -> coref_longformer.CorefLongformerModel:
    """Train model on train dataset until performance no longer improves
    on dev dataset. Return the trained model.
    """
    user = getpass.getuser()
    n_gpus = torch.cuda.device_count()
    devices = list(range(n_gpus))
    device = "cuda:0"
    use_gpu = use_gpu and n_gpus > 0
    use_dynamic_loading = use_dynamic_loading and use_gpu
    n_labels = model.n_labels
    use_data_parallel = use_gpu and use_data_parallel

    if use_official_evaluation:
      logging.info("Config = Using official evaluation")
      assert official_scorer is not None, "Official scorer path missing!"
      assert dev_corpus is not None, "Dev corpus missing!"
      if evaluate_on_train:
        assert train_corpus is not None, "Train corpus missing!"
    
    if evaluate_against_original:
      logging.info("Config = Evaluating against original corpus")
      assert dev_corpus is not None, "Dev corpus missing!"
      if evaluate_on_train:
        assert train_corpus is not None, "Train corpus missing!"
    
    if evaluate_on_train:
      logging.info("Config = Evaluating on train dataset as well")
    
    if use_data_parallel:
      logging.info("Config = Using nn.DataParallel")
      model = nn.DataParallel(model)
    
    if use_dynamic_loading:
      logging.info("Config = Using dynamic loading")
    
    if evaluate_seq:
      logging.info("Config = Evaluating sequence")
    
    if save_model:
      logging.info("Config = Saving best model")
      assert save_directory is not None, "Save directory missing!"
      best_save_directory = os.path.join(save_directory, "best")
      os.makedirs(best_save_directory, exist_ok=True)
    
    if save_predictions:
      logging.info("Config = Saving best predictions")
      assert save_directory is not None, "Save directory missing!"
    
    if save_model_after_every_epoch:
      logging.info("Config = Saving model after every epoch")
      assert save_model, "Set save_model!"
    
    if save_predictions_after_every_epoch:
      logging.info("Config = Saving predictions after every epoch")
      assert save_predictions, "Set save_predictions!"

    train_dataloader = tdata.DataLoader(
      train_dataset, batch_size=train_batch_size, shuffle=True)
    dev_dataloader = tdata.DataLoader(dev_dataset, batch_size=eval_batch_size)
    n_train_batches = len(train_dataloader)
    n_dev_batches = len(dev_dataloader)
    logging.info(f"Train batches = {n_train_batches}")
    logging.info(f"Infer batches = {n_dev_batches}")

    if use_gpu:
      logging.info(f"Loading model to {device}...")
      start_time = time.time()
      model.to(device)
      time_taken = util.convert_float_seconds_to_time_string(
        time.time() - start_time)
      logging.info("...Loading done.")
      logging.info(f"Time taken to load model to {device} = {time_taken}")
    util.print_gpu_usage(user, devices)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
                            weight_decay=weight_decay)
    n_training_steps = max_epochs * n_train_batches
    scheduler = get_linear_schedule_with_warmup(
      optimizer, num_warmup_steps=n_warmup_steps,
      num_training_steps=n_training_steps)
    best_dev_F1 = -0.01
    best_epoch = 0
    epochs_left = patience
    
    logging.info("================================================")
    logging.info("Training started")
    logging.info("================================================")
    for epoch in range(max_epochs):
      model.train()
      logging.info(f"Epoch {epoch + 1}")
      running_batch_loss = []
      running_batch_train_time = []
      epoch_start_time = time.time()
      epoch_predict_ids: list[torch.LongTensor] = []
      epoch_label_ids: list[torch.LongTensor] = []
      epoch_doc_ids: list[torch.IntTensor] = []

      for i, batch in enumerate(train_dataloader):
        (batch_token_ids, batch_mention_ids, batch_label_ids, batch_attn_mask, 
              batch_global_attn_mask, batch_doc_ids) = batch
        
        if use_dynamic_loading:
          batch_token_ids = batch_token_ids.to(device)
          batch_mention_ids = batch_mention_ids.to(device)
          batch_label_ids = batch_label_ids.to(device)
          batch_attn_mask = batch_attn_mask.to(device)
          batch_global_attn_mask = batch_global_attn_mask.to(device)
        
        batch_start_time = time.time()
        optimizer.zero_grad()
        batch_logits = model(batch_token_ids, batch_mention_ids,
          batch_attn_mask, batch_global_attn_mask)
        batch_loss = coref_longformer.compute_loss(batch_logits,
          batch_label_ids, batch_attn_mask, n_labels)
        batch_loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 
          max_grad_norm)
        optimizer.step()
        scheduler.step()
        batch_time_taken = time.time() - batch_start_time
        running_batch_loss.append(batch_loss.detach().item())
        running_batch_train_time.append(batch_time_taken)

        if (i + 1) % print_n_batches == 0:
          average_batch_loss = np.mean(running_batch_loss)
          average_batch_train_time = np.mean(running_batch_train_time)
          estimated_time_remaining = util.convert_float_seconds_to_time_string(
            average_batch_train_time * (n_train_batches - i - 1))
          average_batch_train_time_str = (
            util.convert_float_seconds_to_time_string(
              average_batch_train_time))
          logging.info(f"Batch {i + 1}")
          logging.info("Average training loss @ batch = "
                        f"{average_batch_loss:.4f}")
          logging.info("Average time taken @ batch = "
                        f"{average_batch_train_time_str}")
          logging.info(f"Estimated time remaining for epoch {epoch + 1} = "
                f"{estimated_time_remaining}")
          running_batch_loss = []
          running_batch_train_time = []
        
        batch_predict_ids = batch_logits.argmax(dim=2)
        epoch_label_ids.append(batch_label_ids.cpu())
        epoch_predict_ids.append(batch_predict_ids.detach().cpu())
        epoch_doc_ids.append(batch_doc_ids.cpu())
      
      epoch_time_taken = util.convert_float_seconds_to_time_string(
        time.time() - epoch_start_time)
      logging.info(f"Epoch {epoch + 1} done")
      logging.info(f"Total time taken = {epoch_time_taken}")

      if save_model_after_every_epoch:
        logging.info(f"Saving model after epoch {epoch + 1}")
        epoch_save_directory = os.path.join(
          save_directory, f"epoch_{epoch + 1}")
        os.makedirs(epoch_save_directory, exist_ok=True)
        _model = model.module if use_data_parallel else model
        util.save_model(_model, epoch_save_directory)
      
      epoch_save_directory_train = None
      epoch_save_directory_dev = None
      if save_predictions_after_every_epoch:
        epoch_save_directory_train = os.path.join(save_directory,
          f"epoch_{epoch + 1}/train")
        epoch_save_directory_dev = os.path.join(save_directory,
          f"epoch_{epoch + 1}/dev")
        os.makedirs(epoch_save_directory_train, exist_ok=True)
        os.makedirs(epoch_save_directory_dev, exist_ok=True)

      dev_metric = None
      args = [("dev", dev_corpus, dev_dataloader, epoch_save_directory_dev,
                save_predictions_after_every_epoch)]
      if evaluate_on_train:
        args = [("train", train_corpus, train_dataloader,
                  epoch_save_directory_train,
                  save_predictions_after_every_epoch)] + args
      
      for name, corpus, dataloader, directory, save_flag in args:
        logging.info("================================================")
        logging.info(f"Inference and Evaluation on {name} dataset started")
        logging.info("================================================")
        output = evaluate.evaluate_dataloader(
          model,
          dataloader,
          corpus=corpus,
          official_scorer=official_scorer,
          use_official=use_official_evaluation,
          batch_size=eval_batch_size,
          print_n_batches=print_n_batches,
          use_dynamic_loading=use_dynamic_loading,
          evaluate_against_corpus=evaluate_against_original,
          save_predictions=save_flag,
          predictions_directory=directory,
          evaluate_seq=evaluate_seq
        )
        if evaluate_against_original:
          coref_metric, coref_metric2 = output
        else:
          coref_metric = output
        logging.info(f"Coreference performance on {name} (between tensors)"
                      f" = {coref_metric}")
        if evaluate_against_original:
          logging.info(f"Coreference performance on {name} "
                        f"(between tensor and corpus) = {coref_metric2}")
        if name == "dev":
          dev_metric = coref_metric

      logging.info("================================================")
      logging.info("Checking for early-stopping")
      logging.info("================================================")
      dev_F1 = np.mean(dev_metric.muc.f1 + dev_metric.b3.f1 +
        dev_metric.ceafe.f1)
      delta = 100*(dev_F1 - best_dev_F1)
      logging.info("Average (1/3 x (MUC + B3 + CEAFe)) dev F1 ="
                    f" {100*dev_F1:.1f}")
      if dev_F1 > best_dev_F1:
        if epoch:
          logging.info(f"Dev F1 improved by {delta:.1f}")
        epochs_left = patience
        best_epoch = epoch + 1
        best_dev_F1 = dev_F1
        # TODO: implement saving best model and best predictions
        # You might have to separate inference and evaluation
        # infer(trained_model, dataloader) -> predictions
        # evaluate(predictions, groundtruth) -> CorefMetric
      else:
        logging.info(f"Dev F1 is {-delta:.1f} lower than best Dev F1 "
              f"({100*best_dev_F1:.1f})")
        epochs_left -= 1
      logging.info(f"{epochs_left} epochs left until Dev F1 to improve to "
            "avoid early-stopping!")
      if epochs_left == 0:
        logging.info("Early stopping!")
        break

      logging.info(f"Epoch {epoch + 1} done")
    
    logging.info("================================================")
    logging.info("Training Ended")
    logging.info("================================================")

    logging.info(f"Best Dev F1 = {100*best_dev_F1:.1f}")
    logging.info(f"Best epoch = {best_epoch}")

    return model