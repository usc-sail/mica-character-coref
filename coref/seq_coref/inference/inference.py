"""Functions to run inference"""

from mica_text_coref.coref.seq_coref.models import coref_longformer
from mica_text_coref.coref.seq_coref.utils import util

import logging
import numpy as np
import time
import torch
from torch.utils import data as tdata

def infer(dataloader: tdata.DataLoader,
    model: coref_longformer.CorefLongformerModel,
    batch_size: int,
    use_dynamic_loading = False,
    print_n_batches = 10
    ) -> tuple[torch.LongTensor, torch.LongTensor, torch.FloatTensor,
               torch.IntTensor]:
    """Run inference on the dataloader.
    """
    model.eval()
    device = next(model.parameters()).device
    label_ids_list: list[torch.LongTensor] = []
    prediction_ids_list: list[torch.LongTensor] = []
    doc_ids_list: list[torch.IntTensor] = []
    attn_mask_list: list[torch.FloatTensor] = []
    n_batches = len(dataloader)
    eval_start_time = time.time()
    logging.info(f"Inference batch size = {batch_size}")
    logging.info(f"Number of inference batches = {n_batches}")
    logging.info("Starting inference...")

    with torch.no_grad():
        running_batch_times = []
        for i, batch in enumerate(dataloader):
            (batch_token_ids, batch_mention_ids, batch_label_ids,
             batch_attn_mask, batch_global_attn_mask, batch_doc_ids) = batch

            if use_dynamic_loading:
                batch_token_ids = batch_token_ids.to(device)
                batch_mention_ids = batch_mention_ids.to(device)
                batch_attn_mask = batch_attn_mask.to(device)
                batch_global_attn_mask = batch_global_attn_mask.to(device)

            start_time = time.time()
            batch_logits = model(batch_token_ids, batch_mention_ids,
                                 batch_attn_mask, batch_global_attn_mask)
            batch_prediction_ids = batch_logits.argmax(dim=2)
            label_ids_list.append(batch_label_ids.cpu())
            prediction_ids_list.append(batch_prediction_ids.detach().cpu())
            doc_ids_list.append(batch_doc_ids.cpu())
            attn_mask_list.append(batch_attn_mask.cpu())
            time_taken = time.time() - start_time
            running_batch_times.append(time_taken)

            if (i + 1) % print_n_batches == 0:
                average_time_per_batch = np.mean(running_batch_times)
                estimated_time_remaining = (n_batches - i - 1) * (
                                            average_time_per_batch)
                average_time_per_batch_str = (
                    util.convert_float_seconds_to_time_string(
                        average_time_per_batch))
                estimated_time_remaining_str = (
                    util.convert_float_seconds_to_time_string(
                        estimated_time_remaining))
                time_elapsed_str = util.convert_float_seconds_to_time_string(
                    time.time() - eval_start_time)
                running_batch_times = []

                logging.info(f"Batch {i + 1}")
                logging.info(f"Time elapsed in inference = {time_elapsed_str}")
                logging.info("Average inference time @ batch = "
                             f"{average_time_per_batch_str}")
                logging.info("Estimated inference time remaining = "
                             f"{estimated_time_remaining_str}")

    time_taken = time.time() - eval_start_time
    time_taken_str = util.convert_float_seconds_to_time_string(time_taken)
    logging.info("...Inference done.")
    logging.info(f"Total time taken in inference = {time_taken_str}")

    groundtruth = torch.cat(label_ids_list, dim=0)
    predictions = torch.cat(prediction_ids_list, dim=0)
    doc_ids = torch.cat(doc_ids_list, dim=0)
    attn_mask = torch.cat(attn_mask_list, dim=0)
    return predictions, groundtruth, attn_mask, doc_ids