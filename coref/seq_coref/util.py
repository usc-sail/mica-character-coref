"""Contains utility functions.
"""

from mica_text_coref.coref.seq_coref import data

def find_mention_pair_relationship(
    first_mention: data.Mention, second_mention: data.Mention) -> data.MentionPairRelationship:
    """Find intersection relationship between two mentions"""
    first_mention_elements = set([i for i in range(first_mention.begin, first_mention.end + 1)])
    second_mention_elements = set([i for i in range(second_mention.begin, second_mention.end + 1)])
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
    """Indent and justify block for pretty-printing"""
    block = ""
    current_length_of_line = 0
    n_words_in_current_line = 0
    indent = "".join([" " for _ in range(indent)])
    i = 0

    while i < len(words):
        if current_length_of_line + len(words[i]) < width or n_words_in_current_line == 0:
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