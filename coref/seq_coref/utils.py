"""Contains utility functions.
"""

from mica_text_coref.coref.seq_coref import data

from absl import logging
import contextlib
import gpustat
import os
import time
import torch
from torch import nn

def find_mention_pair_relationship(
    first_mention: data.Mention, second_mention: data.Mention) -> (
    data.MentionPairRelationship):
    """Find intersection relationship between two mentions.

    Args:
        first_mention: First data.Mention object.
        second_mention: Second data.Mention object.
    
    Returns:
        data.MentionPairRelationship, which defines the overlap relationship
        between first_mention and second_mention. It can be EQUAL, DISJOINT,
        SUBSPAN, or INTERSECT
    """
    first_mention_elements = set([i for i in range(
        first_mention.begin, first_mention.end + 1)])
    second_mention_elements = set([i for i in range(
        second_mention.begin, second_mention.end + 1)])
    if first_mention_elements == second_mention_elements:
        return data.MentionPairRelationship.EQUAL
    elif first_mention_elements.isdisjoint(second_mention_elements):
        return data.MentionPairRelationship.DISJOINT
    elif (first_mention_elements.issubset(second_mention_elements) or
     second_mention_elements.issubset(first_mention_elements)):
        return data.MentionPairRelationship.SUBSPAN
    else:
        return data.MentionPairRelationship.INTERSECT

def indent_block(words: list[str], indent: int, width: int) -> str:
    """Indent and justify block for pretty-printing.

    Args:
        words: List of words.
        indent: Indent on the left-hand-side.
        width: Width of the text block.
    
    Returns:
        A text string, indented and justified.
    """
    block = ""
    current_length_of_line = 0
    n_words_in_current_line = 0
    indent = "".join([" " for _ in range(indent)])
    i = 0

    while i < len(words):
        if current_length_of_line + len(words[i]) < width or (
         n_words_in_current_line == 0):
            prefix = " " if n_words_in_current_line > 0 else ""
            block += prefix + words[i]
            current_length_of_line += len(prefix + words[i])
            n_words_in_current_line += 1
            i += 1
        else:
            block += "\n" + indent
            current_length_of_line = 0
            n_words_in_current_line = 0
    
    return block

def convert_float_seconds_to_time_string(seconds: float) -> str:
    """Convert seconds to h m s format"""
    seconds = int(seconds)
    minutes, seconds = seconds//60, seconds%60
    hours, minutes = minutes//60, minutes%60
    return f"{hours}h {minutes}m {seconds}s"

def print_gpu_usage(user: str, devices: list[int]):
    """Print memory consumed on gpus by user processes, and the
    available memory in the gpus.

    Args:
        user: username
        devices: list of gpu device ids
    """
    gpu_collection = gpustat.new_query()
    memory_consumed = 0
    memory_available = 0
    for gpu in gpu_collection.gpus:
        if gpu.index in devices:
            for process in gpu.processes:
                if process["username"] == user:
                    memory_consumed = process["gpu_memory_usage"]
            memory_available = gpu.memory_free
            logging.info(f"GPU {gpu.index} = {memory_consumed} used, "
                    f"{memory_available} free")

def save_model(model: nn.Module, directory: str):
    """Save model's weights to directory with filename `model.pt`.

    Args:
        model: Torch nn.Module
        directory: filepath to directory where model's weights will be saved
    """
    torch.save(model.state_dict(), os.path.join(directory, "model.pt"))

def save_predictions(label_ids: torch.LongTensor,
    prediction_ids: torch.LongTensor,
    doc_ids: torch.IntTensor,
    attn_mask: torch.FloatTensor,
    directory: str):
    """Save label_ids, prediction_ids, doc_ids, and attn_mask tensors to
    directory with name labels.pt, predictions.pt, attn_mask.pt, and doc_ids.pt
    respectively.

    Args:
        label_ids: torch longtensor
        prediction_ids: torch longtensor
        attn_mask: torch float tensor
        doc_ids: torch int tensor
        directory: filepath of directory where the tensors will be saved
    """
    torch.save(label_ids, os.path.join(directory, "labels.pt"))
    torch.save(prediction_ids, os.path.join(directory, "predictions.pt"))
    torch.save(attn_mask, os.path.join(directory, "attn_mask.pt"))
    torch.save(doc_ids, os.path.join(directory, "doc_ids.pt"))

@contextlib.contextmanager
def timer():
    start_time = time.time()
    yield
    time_taken = time.time() - start_time
    time_taken_str = convert_float_seconds_to_time_string(time_taken)
    logging.info(f"Time taken = {time_taken_str}")