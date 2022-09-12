"""Train a coreference longformer model on training dataset until performance
no longer improves on development dataset.
"""

from mica_text_coref.coref.seq_coref import coref_longformer
from mica_text_coref.coref.seq_coref import data
from mica_text_coref.coref.seq_coref import evaluate
from mica_text_coref.coref.seq_coref import util

import numpy as np
import os
import time
import torch
from torch.utils import data as tdata
from torch import optim
from transformers import get_linear_schedule_with_warmup

def train(model: coref_longformer.CorefLongformerModel,
          train_dataset: tdata.TensorDataset,
          dev_dataset: tdata.TensorDataset,
          train_corpus: data.CorefCorpus | None = None,
          dev_corpus: data.CorefCorpus | None = None,
          device = "cpu",
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
          use_dynamic_loading = False
          ) -> coref_longformer.CorefLongformerModel:
    """Train model on train dataset until performance no longer improves
    on dev dataset. Return the trained model.
    """
    user = os.getlogin()
    device_index = -1 if device == "cpu" else int(device.lstrip("cuda:"))

    if use_official_evaluation:
      assert official_scorer is not None and train_corpus is not None and (
        dev_corpus is not None), ("Provide perl scorer script path, train "
          "corpus, and dev corpus if using the official evaluation.")

    train_dataloader = tdata.DataLoader(
      train_dataset, batch_size=train_batch_size, shuffle=True)
    dev_dataloader = tdata.DataLoader(dev_dataset, batch_size=eval_batch_size)
    n_train_batches = len(train_dataloader)
    n_dev_batches = len(dev_dataloader)
    print(f"train batches = {n_train_batches}")
    print(f"eval batches = {n_dev_batches}\n")

    print(f"Loading model to {device}...", end="")
    start_time = time.time()
    model.to(device)
    time_taken = util.convert_float_seconds_to_time_string(
      time.time() - start_time)
    print("done.")
    print(f"Time taken to load model to {device} = {time_taken}")
    if device_index > -1:
      memory_consumed, memory_available = util.get_gpu_usage(user, device_index)
      print(f"GPU memory consumed = {memory_consumed},"
              f" availabe = {memory_available}")
    print()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
                            weight_decay=weight_decay)
    n_training_steps = max_epochs * n_train_batches
    scheduler = get_linear_schedule_with_warmup(
      optimizer, num_warmup_steps=n_warmup_steps,
      num_training_steps=n_training_steps)
    best_dev_F1 = -0.01
    best_epoch = 0
    epochs_left = patience
    
    print("\n\n")
    print("================================================")
    print("Training started")
    print("================================================\n\n")
    for epoch in range(max_epochs):
      model.train()
      print(f"Epoch {epoch + 1}\n")
      running_batch_loss = []
      running_batch_train_time = []
      epoch_start_time = time.time()
      epoch_predict_ids: list[torch.IntTensor] = []
      epoch_label_ids: list[torch.IntTensor] = []
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
        batch_predict_ids, batch_loss = model(
          batch_token_ids, batch_mention_ids, batch_attn_mask,
          batch_global_attn_mask, batch_label_ids)
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
          print(f"Batch {i + 1}")
          print(f"Average training loss @ batch = {average_batch_loss:.4f}")
          print(f"Average time taken @ batch = {average_batch_train_time_str}")
          print(f"Estimated time remaining for epoch {epoch + 1} = "
                f"{estimated_time_remaining}")
          running_batch_loss = []
          running_batch_train_time = []
          print()
        
        epoch_label_ids.append(batch_label_ids.cpu())
        epoch_predict_ids.append(batch_predict_ids.detach().cpu())
        epoch_doc_ids.append(batch_doc_ids.cpu())
      
      epoch_time_taken = util.convert_float_seconds_to_time_string(
        time.time() - epoch_start_time)
      print(f"Epoch {epoch + 1} done")
      print(f"Total time taken = {epoch_time_taken}\n\n")

      if evaluate_on_train:
        print("----------------------------------------------------")
        print("Inference and Evaluation on training dataset started")
        print("----------------------------------------------------\n")
        if use_official_evaluation:
          print("Using the official perl scorer")
        else:
          print("Using the python scorch package")
        print()

        if train_corpus is None:
          train_metric: evaluate.CoreferenceMetric = (
            evaluate.evaluate_dataloader(
              model, train_dataloader, official_scorer=official_scorer, 
              use_official=use_official_evaluation, batch_size=eval_batch_size, 
              print_n_batches=print_n_batches,
              use_dynamic_loading=use_dynamic_loading))
          print("Train Coreference Performance:")
          print(train_metric)
        else:
          train_metric1, train_metric2 = evaluate.evaluate_dataloader(
              model, train_dataloader, corpus=train_corpus, 
              official_scorer=official_scorer, 
              use_official=use_official_evaluation, batch_size=eval_batch_size, 
              print_n_batches=print_n_batches,
              use_dynamic_loading=use_dynamic_loading)
          print("Train Coreference Performance:")
          print(train_metric1)
          print()
          print("Train Coreference Performance (on original corpus):")
          print(train_metric2)
        print("\n")
      
      print("-------------------------------------------------------")
      print("Inference and Evaluation on development dataset started")
      print("-------------------------------------------------------\n")
      if use_official_evaluation:
        print("Using the official perl scorer")
      else:
        print("Using the python scorch package")
      print()

      if dev_corpus is None:
        dev_metric: evaluate.CoreferenceMetric = (
          evaluate.evaluate_dataloader(
            model, dev_dataloader, official_scorer=official_scorer, 
            use_official=use_official_evaluation, batch_size=eval_batch_size, 
            print_n_batches=print_n_batches,
              use_dynamic_loading=use_dynamic_loading))
        print("Dev Coreference Performance:")
        print(dev_metric)
      else:
        dev_metric, dev_metric2 = evaluate.evaluate_dataloader(
            model, dev_dataloader, corpus=dev_corpus, 
            official_scorer=official_scorer, 
            use_official=use_official_evaluation, batch_size=eval_batch_size, 
            print_n_batches=print_n_batches,
              use_dynamic_loading=use_dynamic_loading)
        print("Dev Coreference Performance:")
        print(dev_metric)
        print()
        print("Dev Coreference Performance (on original corpus):")
        print(dev_metric2)
      print("\n")
      
      print("***************************")
      print("Checking for early-stopping")
      print("***************************\n")
      dev_F1 = np.mean(dev_metric.muc.f1 + dev_metric.b3.f1 + 
        dev_metric.ceafe.f1)
      delta = 100*(dev_F1 - best_dev_F1)
      print(f"Average (1/3 x (MUC + B3 + CEAFe)) dev F1 = {100*dev_F1:.1f}")
      if dev_F1 > best_dev_F1:
        if epoch:
          print(f"Dev F1 improved by {delta:.1f}")
        epochs_left = patience
        best_epoch = epoch + 1
        best_dev_F1 = dev_F1
      else:
        print(f"Dev F1 is {-delta:.1f} lower than best Dev F1 "
              f"({100*best_dev_F1:.1f})")
        epochs_left -= 1
      print(f"{epochs_left} epochs left until Dev F1 to improve to "
            "avoid early-stopping!")
      if epochs_left == 0:
        print("Early stopping!")
        break

      print(f"Epoch {epoch + 1} done\n\n")
    
    print("================================================")
    print("Training Ended")
    print("================================================\n\n")

    print(f"Best Dev F1 = {100*best_dev_F1:.1f}")
    print(f"Best epoch = {best_epoch}")

    return model