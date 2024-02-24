"""Evaluate coreference using official conll-2012 scripts"""
from .data import CorefDocument

from collections import defaultdict
import re
import subprocess
import os
import hashlib
import json
    
def convert_to_conll(document: CorefDocument, clusters: list[set[tuple[int, int]]]) -> list[str]:
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
    non_unigram_mentions_sorted_by_begin = sorted(non_unigram_mentions, key=lambda mention: (mention[0], -mention[1]))
    non_unigram_mentions_sorted_by_end = sorted(non_unigram_mentions, key=lambda mention: (mention[1], -mention[0]))

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

def evaluate_conll(reference_scorer: str, gold_conll_lines: list[str], pred_conll_lines: list[str], gold_file: str,
                   pred_file: str) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Evaluate coreference using conll reference scorer. gold_conll_lines contain the gold annotations, and
    pred_conll_lines contain the predicted labels.

    Args:
        reference_scorer: Path to the perl scorer.
        gold_conll_lines: List of lines in conll-format containing gold labels.
        pred_conll_lines: List of lines in conll-format containing predicted labels.
        gold_file: File to which the gold_conll_lines will be saved.
        pred_file: File to which the pred_conll_lines will be saved.

    Returns:
        Dictionary with metric names as keys: muc, bcubed, ceafe. Values are dictionaries with movie names as keys +
            "all" for micro-averaged values. Values of this inner movie-level
        dictionary is a tuple of 3 floats: precision, recall, f1.
    """
    score_pattern = re.compile(
        r"Recall: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\s+Precision: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\s+F1: ([0-9.]+)%")
    document_pattern = re.compile(r"====> (\w+);")
    metric_pattern = re.compile(r"METRIC (\w+):")
    pattern = re.compile(f"({score_pattern.pattern})|({document_pattern.pattern})|({metric_pattern.pattern})")
    with open(gold_file, "w") as f1, open(pred_file, "w") as f2:
        f1.writelines(gold_conll_lines)
        f2.writelines(pred_conll_lines)
    os.makedirs(os.path.join(os.getenv("DATA_DIR"), ".conll"), exist_ok=True)
    with open(gold_file, "rb") as f1, open(pred_file, "rb") as f2:
        bytes1, bytes2 = f1.read(), f2.read()
        hash1, hash2 = hashlib.md5(bytes1).hexdigest(), hashlib.md5(bytes2).hexdigest()
        file_ = os.path.join(os.getenv("DATA_DIR"), ".conll", f"{hash1}_{hash2}.json")
        if os.path.exists(file_):
            with open(file_) as fr:
                result = json.load(fr)
        else:
            cmd = [reference_scorer, "conll", gold_file, pred_file]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            stdout, _ = process.communicate()
            process.wait()
            stdout = stdout.decode("utf-8")
            result = defaultdict(lambda: defaultdict(tuple[float, float, float]))
            metric, movie, n_score_pattern_rows = "", "", 0
            n_movies = len(set(map(lambda match: match.group(1), document_pattern.finditer(stdout))))
            for match in pattern.finditer(stdout):
                if match.group(1):
                    n_score_pattern_rows += 1
                    recall = float(match.group(2))
                    precision = float(match.group(3))
                    f1 = float(match.group(4))
                    if n_score_pattern_rows <= n_movies:
                        result[metric][movie] = (precision, recall, f1)
                    elif n_score_pattern_rows == n_movies + 2:
                        result[metric]["all"] = (precision, recall, f1)
                elif match.group(5):
                    movie = match.group(6)
                else:
                    metric = match.group(8)
                    n_score_pattern_rows = 0
            write_result = {}
            for metric, movie_metric in result.items():
                write_result[metric] = {}
                for movie, metrics in movie_metric.items():
                    write_result[metric][movie] = list(metrics)
            with open(file_, "w") as fw:
                json.dump(write_result, fw)
    return result