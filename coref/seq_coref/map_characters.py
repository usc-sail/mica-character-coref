"""Functions to map two charater strings.

The first character string called word_characters is a concatenation of all
words in a document. The second character string called token_characters is
a concatenation of all tokens which are obtained from the tokenization of the
document using a transformer-based tokenizer.
"""

import numpy as np
import tqdm

def map_characters_naive(word_characters: str, token_characters: str, 
        ignore_token_characters: list[str] = []) -> list[int]:
    """Find the mapping between the word_characters and token_characters
    strings. This function assumes every word character is mapped to some
    token character.

    Args:
        word_characters: A text string of word characters.
        token_characters: A text string of token characters.
        ignore_token_characters: A list of token charaters to ignore during
            mapping.
    
    Returns:
        An integer list of the same length as word_characters.
    """
    word_character_to_token_character = np.zeros(len(word_characters),
                                                dtype=int)
    i, j = 0, 0
    while i < len(word_characters) and j < len(token_characters):
        equal = word_characters[i] == token_characters[j] and (
                token_characters[j] not in ignore_token_characters)
        if equal:
            word_character_to_token_character[i] = j
            i += 1
            j += 1
        else:
            j += 1
    return word_character_to_token_character.tolist()

def map_characters_lcs(word_characters: str, token_characters: str, 
        ignore_token_characters: list[str] = []) -> list[int]:
    """Find the mapping between the word_characters and token_characters
    strings using longest common subsequence algorithm. Some word characters
    might not be matched using this algorithm.

    Args:
        word_characters: A text string of word characters.
        token_characters: A text string of token characters.
        ignore_token_characters: A list of token charaters to ignore during
            mapping.
    
    Returns:
        An integer list of the same length as word_characters.
    """
    lcs_len = np.zeros((len(word_characters), len(token_characters)), dtype=int)
    lcs_dir = np.zeros((len(word_characters), len(token_characters)), dtype=int)

    for i in tqdm.trange(len(word_characters)):
        for j in range(len(token_characters)):
            equal = word_characters[i] == token_characters[j] and (
                token_characters[j] not in ignore_token_characters)
            if i == 0 and j == 0:
                if equal:
                    lcs_len[i, j] = 1
                    lcs_dir[i, j] = 1
                else:
                    lcs_len[i, j] = 0
                    lcs_dir[i, j] = 0
            elif i == 0:
                if equal:
                    lcs_len[i, j] = 1
                    lcs_dir[i, j] = 1
                else:
                    lcs_len[i, j] = lcs_len[i, j - 1]
                    lcs_dir[i, j] = 2
            elif j == 0:
                if equal:
                    lcs_len[i, j] = 1
                    lcs_dir[i, j] = 1
                else:
                    lcs_len[i, j] = lcs_len[i - 1, j]
                    lcs_dir[i, j] = 3
            else:
                if equal:
                    lcs_len[i, j] = 1 + lcs_len[i - 1, j - 1]
                    lcs_dir[i, j] = 1
                else:
                    if lcs_len[i, j - 1] > lcs_len[i - 1, j]:
                        lcs_len[i, j] = lcs_len[i, j - 1]
                        lcs_dir[i, j] = 2
                    else:
                        lcs_len[i, j] = lcs_len[i - 1, j]
                        lcs_dir[i, j] = 3

    word_character_to_token_character = np.zeros(len(word_characters), 
                                                dtype=int)
    i = len(word_characters) - 1
    j = len(token_characters) - 1
    while i >= 0 and j >= 0:
        if lcs_dir[i, j] == 1:
            word_character_to_token_character[i] = j
            i -= 1
            j -= 1
        elif lcs_dir[i, j] == 2:
            j -= 1
        elif lcs_dir[i, j] == 3:
            i -= 1
        else:
            i -= 1
            j -= 1

    i = 0
    while i < len(word_character_to_token_character):
        if i > 0 and word_character_to_token_character[i - 1] > 0 and (
            word_character_to_token_character[i] == 0):
            j = i
            while j < len(word_character_to_token_character) and (
                word_character_to_token_character[j] == 0):
                word_character_to_token_character[j] = (
                    word_character_to_token_character[i - 1])
                j += 1
            i = j
        else:
            i += 1

    return word_character_to_token_character.tolist()