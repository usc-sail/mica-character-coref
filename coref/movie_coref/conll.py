"""Evaluate coreference using official conll-2012 scripts
"""

from mica_text_coref.coref.movie_coref.data import CorefDocument

import re
import subprocess

def convert_to_conll(
    document: CorefDocument, clusters: list[set[tuple[int, int]]]) -> list[str]:
    """Create conll lines from clusters.

    Args:
        document: CorefDocument object representing the movie.
        clusters: List of sets. Each set is a set of integer tuples.
            Each integer tuple contains two integers: start and end of the
            mention

    Returns:
        List of lines in conll-format. Each line contains the word and
        coreference tag.
    """
    total_n_tokens = len(document.token)
    coref_column = ["-" for _ in range(total_n_tokens)]
    mentions = [(j, k, i + 1) for i, cluster in enumerate(clusters) for j, k in cluster]
    non_unigram_mentions = list(filter(lambda mention: mention[1] - mention[0] > 0, mentions))
    unigram_mentions = list(filter(lambda mention: mention[1] == mention[0], mentions))
    non_unigram_mentions_sorted_by_begin = sorted(non_unigram_mentions, key=lambda mention: 
        (mention[0], -mention[1]))
    non_unigram_mentions_sorted_by_end = sorted(non_unigram_mentions, key=lambda mention: 
        (mention[1], -mention[0]))

    for begin, _, cluster_index in non_unigram_mentions_sorted_by_begin:
        if coref_column[begin] == "-":
            coref_column[begin] = "(" + str(cluster_index)
        else:
            coref_column[begin] += "|(" + str(cluster_index)

    for begin, _, cluster_index in unigram_mentions:
        if coref_column[begin] == "-":
            coref_column[begin] = "(" + str(cluster_index) + ")"
        else:
            coref_column[begin] += "|(" + str(cluster_index) + ")"

    for _, end, cluster_index in non_unigram_mentions_sorted_by_end:
        if coref_column[end] == "-":
            coref_column[end] = str(cluster_index) + ")"
        else:
            coref_column[end] += "|" + str(cluster_index) + ")"

    split_parts = document.movie.split("_")
    if re.match(r"^\d+$", split_parts[-1]) is None:
        movie, part = document.movie, "00"
    else:
        movie, part = "_".join(split_parts[:-1]), split_parts[-1].zfill(2)

    conll_lines = [f"#begin document {movie}; part {part}\n"]
    max_word_width = max(len(token) for token in document.token)
    max_speaker_width = max(len(speaker) for speaker in document.speaker)
    for j in range(total_n_tokens):
        part_integer = int(part)
        word = document.token[j]
        pos = document.pos[j]
        parse = document.parse[j]
        ner = document.ner[j]
        speaker = document.speaker[j]
        is_pronoun = int(document.is_pronoun[j])
        is_punct = int(document.is_punctuation[j])
        coref = coref_column[j]
        elements = [
            f"{movie:15s}", 
            f"{part_integer:2d}",
            f"{j:6d}",
            f"{word:{max_word_width}s}",
            f"{pos:5s}",
            f"{parse:3s}",
            f"{ner:10s}",
            f"{speaker:{max_speaker_width}s}",
            f"{is_pronoun:2d}",
            f"{is_punct:2d}",
            coref
        ]
        line = "\t".join(elements) + "\n"
        conll_lines.append(line)
    conll_lines.append("\n")
    conll_lines = conll_lines[:-1] + ["#end document\n"]

    return conll_lines

def evaluate_conll(reference_scorer: str, gold_conll_lines: list[str],
    pred_conll_lines: list[str], gold_file: str, pred_file: str) -> list[float]:
    """Evaluate coreference using conll reference scorer. gold_conll_lines
    contain the gold annotations, and pred_conll_lines contain the predicted
    labels.

    Args:
        reference_scorer: Path to the perl scorer.
        gold_conll_lines: List of lines in conll-format containing gold labels.
        pred_conll_lines: List of lines in conll-format containing predicted
            labels.
        gold_file: File to which the gold_conll_lines will be saved.
        pred_file: File to which the pred_conll_lines will be saved.

    Returns:
        List of 6 numbers in the following order: precision and recall of
        MUC, precision and recall of B-cubed, precision and recall of CEAF-e.
    """
    with open(gold_file, "w") as f1, open(pred_file, "w") as f2:
        f1.writelines(gold_conll_lines)
        f2.writelines(pred_conll_lines)
    values = []
    for metric in ["muc", "bcub", "ceafe"]:
        cmd = [reference_scorer, metric, gold_file, pred_file, "none"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        stdout, _ = process.communicate()
        process.wait()
        stdout = stdout.decode("utf-8")
        matched_tuples = re.findall(r"Coreference: Recall: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\s+"
            r"Precision: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\s+F1: ([0-9.]+)%", stdout, flags=re.DOTALL)
        recall = float(matched_tuples[0][0])/100
        precision = float(matched_tuples[0][1])/100
        values.extend([precision, recall])
    return values