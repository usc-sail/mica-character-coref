"""Functions to run inference"""

from mica_text_coref.coref.seq_coref import acceleration
from mica_text_coref.coref.seq_coref import coref_longformer
from mica_text_coref.coref.seq_coref import utils

import numpy as np
import time
import torch
from torch.utils import data as tdata

def infer(dataloader: tdata.DataLoader,
          model: coref_longformer.CorefLongformerModel,
          print_n_batches=10
          ) -> tuple[torch.LongTensor, torch.LongTensor, torch.FloatTensor,
                     torch.IntTensor]:
    """Run inference on the dataloader.

    Args:
        dataloader: PyTorch dataloader.
        model: Longformer coreference model.
        print_n_batches: Number of batches after which to log.
    
    Returns:
        A tuple of four tensors, as follows:
            label_ids: LongTensor of label ids.
            prediction_ids: LongTensor of prediction ids.
            attn_mask: FloatTensor of attention mask.
            doc_ids: IntTensor of document ids.
    """
    # Initialize variables
    # device = torch.device("cuda:0")
    accelerator, logger = acceleration.accelerator, acceleration.logger
    model.eval()
    label_ids_list: list[torch.LongTensor] = []
    prediction_ids_list: list[torch.LongTensor] = []
    doc_ids_list: list[torch.IntTensor] = []
    attn_mask_list: list[torch.FloatTensor] = []
    n_batches = len(dataloader)
    logger.info(f"Number of inference batches = {n_batches}")

    # Inference Loop
    with utils.timer("inference"), torch.no_grad():
        running_batch_times = []
        for i, batch in enumerate(dataloader):

            # One inference step
            (batch_token_ids, batch_mention_ids, batch_label_ids,
             batch_attn_mask, batch_global_attn_mask, batch_doc_ids) = batch
            # batch_token_ids = batch_token_ids.to(device)
            # batch_mention_ids = batch_mention_ids.to(device)
            # batch_label_ids = batch_label_ids.to(device)
            # batch_attn_mask = batch_attn_mask.to(device)
            # batch_global_attn_mask = batch_global_attn_mask.to(device)
            # batch_doc_ids = batch_doc_ids.to(device)
            start_time = time.time()
            batch_logits = model(batch_token_ids, batch_mention_ids,
                                 batch_attn_mask, batch_global_attn_mask)
            batch_logits, batch_label_ids = accelerator.gather_for_metrics(
                (batch_logits, batch_label_ids))
            batch_prediction_ids = batch_logits.argmax(dim=2)
            label_ids_list.append(batch_label_ids)
            prediction_ids_list.append(batch_prediction_ids)
            doc_ids_list.append(batch_doc_ids)
            attn_mask_list.append(batch_attn_mask)
            time_taken = time.time() - start_time
            running_batch_times.append(time_taken)

            # Log after print_n_batches
            if (i + 1) % print_n_batches == 0:
                average_time_per_batch = np.mean(running_batch_times)
                estimated_time_remaining = (n_batches - i - 1) * (
                                            average_time_per_batch)
                average_time_per_batch_str = (
                    utils.convert_float_seconds_to_time_string(
                        average_time_per_batch))
                estimated_time_remaining_str = (
                    utils.convert_float_seconds_to_time_string(
                        estimated_time_remaining))
                running_batch_times = []

                logger.info(f"Batch {i + 1}")
                logger.info("Average inference time @ batch = "
                             f"{average_time_per_batch_str}")
                logger.info("Estimated inference time remaining = "
                             f"{estimated_time_remaining_str}")

    # Concat batch outputs
    groundtruth = torch.cat(label_ids_list, dim=0)
    predictions = torch.cat(prediction_ids_list, dim=0)
    doc_ids = torch.cat(doc_ids_list, dim=0)
    attn_mask = torch.cat(attn_mask_list, dim=0)
    return predictions, groundtruth, attn_mask, doc_ids