"""Contains utility functions.
"""

from mica_text_coref.coref.seq_coref import data

import gpustat

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

def get_gpu_usage(user: str, device: int) -> tuple[int, int]:
    """Find memory consumed on gpu by user processes, and available memory"""
    gpu_collection = gpustat.new_query()
    memory_consumed = 0
    memory_available = 0
    for gpu in gpu_collection.gpus:
        if gpu.index == device:
            for process in gpu.processes:
                if process["username"] == user:
                    memory_consumed = process["gpu_memory_usage"]
            memory_available = gpu.memory_free
            return memory_consumed, memory_available
    assert False, f"GPU:{device} not found"