"""Evaluator class for coreference resolution and character head prediction."""
from . import conll
from . import data
from . import rules

import collections
import json
import math
import numpy as np
import os
import re
from scipy.optimize import linear_sum_assignment
import subprocess
import tempfile
import typing

Cluster = set[typing.Hashable]
SpanCluster = set[tuple[int, int]]

class Evaluator:
    """Evaluate coreference resolution and character head prediction, and cache results."""

    def __init__(self, cache_file: str = None) -> None:
        """Initialize coreference and character head prediction evaluator.

        Args:
            cache_file (str): File where coreference results are cached. If not provided, evaluator does not cache
                results.
        """
        self._cache = None
        if cache_file is not None:
            if not os.path.exists(cache_file):
                self._cache = {}
            else:
                with open(cache_file, "r") as f:
                    self._cache = json.load(f)
    
    def _hash(self, key: list[Cluster], response: list[Cluster]) -> int:
        """Hash key and response clusters so that they can be added as keys to cache.
        
        Args:
            key (list[set]): List of key clusters.
            response (list[set]): List of response clusters.
        
        Return:
            Text string representing the hash of the key and response clusters.
        """
        hash_elements = [len(key), len(response)]
        key = sorted(key, key=lambda cluster: len(cluster), reverse=True)
        response = sorted(response, key=lambda cluster: len(cluster), reverse=True)
        for clusters in [key, response]:
            for cluster in clusters:
                hash_elements.append(len(cluster))
        for key_cluster in key:
            for response_cluster in response:
                hash_elements.append(len(key_cluster.intersection(response_cluster)))
        return hash(tuple(hash_elements))
    
    def _read_cache(self, hash: int, metric: str) -> tuple[float, float, float, float] | None:
        """Read metric result from cache. If the hash key is not present in the cache or cache file is not provided
        , return None.

        Args:
            hash (int): Hash of the key and response clusters.
            metric (str): Metric name.
        
        Return:
            Tuple of four numbers - recall numerator, recall denominator, precision numerator, and precision denominator
        """
        if self._cache is not None:
            try:
                return self._cache[hash][metric]
            except KeyError: pass
        return None
    
    def _write_cache(self, hash: int, metric: str, scores: tuple[float, float, float, float]) -> None:
        """Write metric result to cache.

        Args:
            hash (int): Hash of the key and response clusters.
            metric (str): Metric name.
            scores (tuple): recall numerator, recall denominator, precision numerator, and precision denominator.
        """
        if self._cache is not None:
            if hash not in self._cache:
                self._cache[hash] = {}
            self._cache[hash][metric] = list(scores)
    
    def _trace(self, cluster: Cluster, other_clusters: list[Cluster]):
        """Generate the partitions of cluster created by other_clusters.
        
        Args:
            cluster (set): Cluster.
            other_clusters (list[set]): List of other clusters.
        
        Return:
            Generator of cluster in the partition.
        """
        remaining = set(cluster)
        for a in other_clusters:
            common = remaining.intersection(a)
            if common:
                remaining.difference_update(common)
                yield common
        for x in sorted(remaining):
            yield set((x,))

    def _muc(self, key: list[Cluster], response: list[Cluster]) -> tuple[float, float, float, float]:
        """Calculate MUC metric from key and response clusters.

        Args:
            key (list[set]): List of key clusters.
            response (list[set]): List of response clusters.
        
        Return:
            Tuple of four numbers - recall numerator, recall denominator, precision numerator, and precision denominator
        """
        if all(len(k) == 1 for k in key) or all(len(r) == 1 for r in response):
            return 0.0, 0.0, 0.0, 0.0
        R_numer = sum(len(k) - sum(1 for _ in self._trace(k, response)) for k in key)
        R_denom = sum(len(k) - 1 for k in key)
        P_numer = sum(len(r) - sum(1 for _ in self._trace(r, key)) for r in response)
        P_denom = sum(len(r) - 1 for r in response)
        return R_numer, R_denom, P_numer, P_denom

    def _bcub(self, key: list[Cluster], response: list[Cluster]) -> tuple[float, float, float, float]:
        """Calculate B-cube metric from key and response clusters.

        Args:
            key (list[set]): List of key clusters.
            response (list[set]): List of response clusters.
        
        Return:
            Tuple of four numbers - recall numerator, recall denominator, precision numerator, and precision denominator
        """
        if sum(len(k) for k in key) == 0:
            R_numer, R_denom = 0.0, 0.0
        else:
            R_numer = math.fsum(len(k.intersection(r)) ** 2 / len(k) for k in key for r in response)
            R_denom = sum(len(k) for k in key)
        if sum(len(r) for r in response) == 0:
            P_numer, P_denom = 0.0, 0.0
        else:
            P_numer = math.fsum(len(r.intersection(k)) ** 2 / len(r) for r in response for k in key)
            P_denom = sum(len(r) for r in response)
        return R_numer, R_denom, P_numer, P_denom

    def _ceaf(self, key: list[Cluster], response: list[Cluster],
              score: typing.Callable[[Cluster, Cluster], float]) -> tuple[float, float, float, float]:
        """Helper function to calculate CEAF-based metrics.
        
        Args:
            key (list[set]): List of key clusters.
            response (list[set]): List of response clusters.
            score ((set, set) -> float): Callable that scores the alignment between two clusters.
        
        Return:
            Tuple of four numbers - recall numerator, recall denominator, precision numerator, and precision denominator
        """
        if len(response) == 0 or len(key) == 0:
            return 0.0, 0.0, 0.0, 0.0
        else:
            cost_matrix = np.array([[-score(k, r) for r in response] for k in key])
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            total_score = -cost_matrix[row_ind, col_ind].sum()
            R_numer = P_numer = total_score
            R_denom = math.fsum(score(k, k) for k in key)
            P_denom = math.fsum(score(r, r) for r in response)
            return R_numer, R_denom, P_numer, P_denom

    def _ceafm(self, key: list[Cluster], response: list[Cluster]) -> tuple[float, float, float, float]:
        """Calculate CEAF-entity metric from key and response clusters.

        Args:
            key (list[set]): List of key clusters.
            response (list[set]): List of response clusters.
        
        Return:
            Tuple of four numbers - recall numerator, recall denominator, precision numerator, and precision denominator
        """
        def Φ_3(k, r):
            return len(k.intersection(r))
        return self._ceaf(key, response, Φ_3)

    def _ceafe(self, key: list[Cluster], response: list[Cluster]) -> tuple[float, float, float, float]:
        """Calculate CEAF-mention metric from key and response clusters.

        Args:
            key (list[set]): List of key clusters.
            response (list[set]): List of response clusters.
        
        Return:
            Tuple of four numbers - recall numerator, recall denominator, precision numerator, and precision denominator
        """
        def Φ_4(k, r):
            return 2 * len(k.intersection(r)) / (len(k) + len(r))
        return self._ceaf(key, response, Φ_4)
    
    def _lea(self, key: list[Cluster], response: list[Cluster]) -> tuple[float, float, float, float]:
        """Calculate LEA metric from key and response clusters.

        Args:
            key (list[set]): List of key clusters.
            response (list[set]): List of response clusters.
        
        Return:
            Tuple of four numbers - recall numerator, recall denominator, precision numerator, and precision denominator
        """
        key_importance = np.array([len(key_) for key_ in key])
        response_importance = np.array([len(response_) for response_ in response])
        assert np.all(key_importance > 0), "Empty cluster in key"
        assert np.all(response_importance > 0), "Empty cluster in response"
        intersection_counts = np.zeros((len(key), len(response)), dtype=int)
        for i, key_ in enumerate(key):
            for j, response_ in enumerate(response):
                intersection_counts[i, j] = len(key_.intersection(response_))
        link = intersection_counts * (intersection_counts - 1) / 2
        singleton = (key_importance == 1).reshape(-1, 1) & (response_importance == 1).reshape(1, -1)
        link[singleton] = intersection_counts[singleton]
        key_link = np.maximum(key_importance * (key_importance - 1) / 2, 1)
        response_link = np.maximum(response_importance * (response_importance - 1) / 2, 1)
        recall_numer = (key_importance * link.sum(axis=1) / key_link).sum()
        recall_denom = key_importance.sum()
        precision_numer = (response_importance * link.sum(axis=0) / response_link).sum()
        precision_denom = response_importance.sum()
        return recall_numer, recall_denom, precision_numer, precision_denom

    def _run_scorer(self, key_file: str, response_file: str, scorer_path: str) -> (
            dict[str, dict[str, tuple[float, float, float, float]]]):
        """Calculate CONLL metrics (MUC, B-cube, and CEAF-e) by running the perl scorer.

        Args:
            scorer_path (str): Path to the perl scorer.
            key_file (str): Path to the CONLL file containing key clusters.
            response_file (str): Path to the CONLL file containing response clusters.
        
        Return:
            Dictionary of movie to metric to four numbers - recall numerator, recall denominator, precision numerator,
            and precision denominator.
        """
        cmd = [scorer_path, "conll", key_file, response_file]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        stdout, _ = process.communicate()
        process.wait()
        stdout = stdout.decode("utf-8")
        score_pattern = re.compile(r"\nRecall: \(([0-9.]+) / ([0-9.]+)\) [0-9.]+%\s+"
                               r"Precision: \(([0-9.]+) / ([0-9.]+)\) [0-9.]+%\s+F1: [0-9.]+%")
        document_pattern = re.compile(r"====> (\w+); part (\d+):")
        metric_pattern = re.compile(r"METRIC (\w+):")
        pattern = re.compile(f"({score_pattern.pattern})|({document_pattern.pattern})|({metric_pattern.pattern})")
        result = {}
        metric, movie, n_score_pattern_rows = "", "", 0
        for match in pattern.finditer(stdout):
            if match.group(1):
                n_score_pattern_rows += 1
                recall_numer = float(match.group(2))
                recall_denom = float(match.group(3))
                precision_numer = float(match.group(4))
                precision_denom = float(match.group(5))
                if movie not in result:
                    result[movie] = {}
                result[movie][metric] = (recall_numer, recall_denom, precision_numer, precision_denom)
            elif match.group(6):
                movie = match.group(7)
                part = int(match.group(8))
                if part > 0:
                    movie = f"{movie}_{part}"
            elif match.group(9):
                metric = match.group(10)
        return result

    def _seq(self, key_seq: list[int], response_seq: list[int], pos_class: int) -> tuple[float, float, float, float]:
        """Compare two sequences and calculate precision and recall for the positive class.

        Args:
            key_seq (list): List of key values.
            response_seq (list): List of response values.
        
        Return:
            Tuple of four numbers - recall numerator, recall denominator, precision numerator, and precision denominator
        """
        key_seq_ = np.array(key_seq)
        response_seq_ = np.array(response_seq)
        tp = ((key_seq_ == pos_class) & (response_seq_ == pos_class)).sum()
        fp = ((key_seq_ != pos_class) & (response_seq_ == pos_class)).sum()
        fn = ((key_seq_ == pos_class) & (response_seq_ != pos_class)).sum()
        return tp, tp + fn, tp, tp + fp
    
    def pycoref(self, movie_to_key: dict[str, list[Cluster]], movie_to_response: dict[str, list[Cluster]],
                metrics: list[str] = ["muc", "bcub", "ceafe", "lea"]) -> dict[str, dict[str, data.Metric]]:
        """Calculate coreference metrics in the `metrics` list for each movie and their micro-averaged score, using
        python implementations (not the reference scorer).

        Args:
            movie_to_key (dict[str, list[Cluster]]): Dictionary of movie name to key clusters.
            movie_to_response (dict[str, list[Cluster]]): Dictionary of movie name to response clusters.
            metrics (list[str]): Metrics to calculate. Metrics can be "muc", "bcub", "ceafe", or "lea".

        Return:
            Dictionary of movie name to metric name to Metric object. The micro-averaged score can be found in the
            "micro" movie name.
        """
        assert set(metrics).issubset(["muc", "bcub", "ceafe", "lea"]), "metrics can be muc, bcub, ceafe, or lea"
        movie_to_metric: dict[str, dict[str, data.Metric]] = {}
        metric_to_scores = collections.defaultdict(lambda: np.zeros(4))
        for movie, key in movie_to_key.items():
            if movie not in movie_to_response:
                continue
            response = movie_to_response[movie]
            movie_to_metric[movie] = {}
            for metric in metrics:
                if metric == "muc":
                    sample_score = self._muc(key, response)
                elif metric == "bcub":
                    sample_score = self._bcub(key, response)
                elif metric == "ceafe":
                    sample_score = self._ceafe(key, response)
                else:
                    sample_score = self._lea(key, response)
                metric_to_scores[metric] += sample_score
                recall = sample_score[0]/(1e-23 + sample_score[1])
                precision = sample_score[2]/(1e-23 + sample_score[3])
                movie_to_metric[movie][metric] = data.Metric(recall, precision)
        if movie_to_metric:
            movie_to_metric["micro"] = {}
            for metric, micro_scores in metric_to_scores.items():
                recall = micro_scores[0]/(1e-23 + micro_scores[1])
                precision = micro_scores[2]/(1e-23 + micro_scores[3])
                movie_to_metric["micro"][metric] = data.Metric(recall, precision)
        return movie_to_metric
    
    def perlcoref(self, key_file: str, response_file: str, scorer_path: str) -> dict[str, dict[str, data.Metric]]:
        """Calculate MUC, Bcub, and CEAFe metrics, and their micro-averaged score by running the perl scorer on the
        `key_file` and `response_file`.

        Args:
            key_file (str): Key CONLL file.
            response_file (str): Response CONLL file.
            scorer_path (str): Path to scorer.
        
        Return:
            Dictionary of movie name to metric name to Metric object. The micro-averaged score can be found in the
            "micro" movie name.
        """
        movie_to_metric_scores = self._run_scorer(key_file, response_file, scorer_path)
        movie_to_metric: dict[str, dict[str, data.Metric]] = {}
        metric_to_scores = collections.defaultdict(lambda: np.zeros(4))
        for movie, metric_dict in movie_to_metric_scores.items():
            movie_to_metric[movie] = {}
            for metric, scores in metric_dict.items():
                recall = scores[0]/(1e-23 + scores[1])
                precision = scores[2]/(1e-23 + scores[3])
                movie_to_metric[movie][metric] = data.Metric(recall, precision)
                metric_to_scores[metric] += scores
        if movie_to_metric:
            movie_to_metric["micro"] = {}
            for metric, micro_scores in metric_to_scores.items():
                recall = micro_scores[0]/(1e-23 + micro_scores[1])
                precision = micro_scores[2]/(1e-23 + micro_scores[3])
                movie_to_metric["micro"][metric] = data.Metric(recall, precision)
        return movie_to_metric
    
    def evaluate(self, documents: list[data.CorefDocument], results: list[data.CorefResult], scorer_path: str = None, 
                 only_lea: bool = False, remove_speaker_links: bool = False, output_dir: str = None,
                 filename: str = "unk") -> data.MovieCorefMetric:
        """Evaluate coreference resolution and character head prediction.
    
        If scorer is provided, then it used to calculate the CONLL metrics, otherwise pythonic functions are used.

        CONLL files are written to 'output_dir' with 'filename' included in the file names.
        If 'output_dir' is NONE, then temporary directories are used that are immediately deleted.

        Args:
            documents (list): List of CorefDocument objects. Used to obtain key clusters and character head values.
            results (list): List of CorefResult objects. Used to obtain the response clusters and character heads.
            scorer_path (str): Filepath of the official perl conll-2012 scorer.
            only_lea (bool): Evaluate only LEA metric and don't evaluate CONLL metrics.
            remove_speaker_links (bool): Remove coreference links that include speaker from both key and response
                clusters. Done for fair comparison between preprocess == "nocharacters" and
                preprocess == "addsays" or "none".
            output_dir (str): Directory where CONLL files are written. If NONE, temporary directories are used.
            filename (str): Text to be included in the CONLL filenames.
        
        Returns:
            MovieCorefMetric
        """
        assert len(documents) == len(results), "documents and results should be of same length"
        
        # Initialize key and predicted variables
        movie_to_key_word_clusters, movie_to_response_word_clusters = {}, {}
        movie_to_key_span_clusters, movie_to_response_span_clusters = {}, {}
        key_character_seq, response_character_seq = [], []

        for document, result in zip(documents, results):
            # Get key and response clusters for each document
            key_word_clusters = [set((mention.head, mention.head) for mention in cluster)
                                        for cluster in document.clusters.values()]
            key_span_clusters = [set((mention.begin, mention.end) for mention in cluster)
                                        for cluster in document.clusters.values()]
            response_word_clusters = [set((word, word) for word in cluster)
                                        for cluster in result.predicted_word_clusters]
            response_span_clusters = result.predicted_span_clusters

            # Remove speaker links for preprocess != "nocharacters"
            if remove_speaker_links:
                key_word_clusters = rules.remove_speaker_links(key_word_clusters, document.parse)
                key_span_clusters = rules.remove_speaker_links(key_span_clusters, document.parse)
                response_word_clusters = rules.remove_speaker_links(response_word_clusters, document.parse)
                response_span_clusters = rules.remove_speaker_links(response_span_clusters, document.parse)
            
            movie_to_key_word_clusters[document.movie] = key_word_clusters
            movie_to_response_word_clusters[document.movie] = response_word_clusters
            movie_to_key_span_clusters[document.movie] = key_span_clusters
            movie_to_response_span_clusters[document.movie] = response_span_clusters

            # Get key and response character head sequences
            key_character_seq.extend(document.word_head_ids)
            response_character_seq.extend(result.predicted_character_heads.tolist())

        # calculate LEA
        movie_coref_metric = data.MovieCorefMetric()
        movie_coref_metric.word_coref.lea = self.pycoref(movie_to_key_word_clusters, movie_to_response_word_clusters,
                                                         metrics=["lea"])["micro"]["lea"]
        movie_coref_metric.span_coref.lea = self.pycoref(movie_to_key_span_clusters, movie_to_response_span_clusters,
                                                         metrics=["lea"])["micro"]["lea"]

        # calculate character head prediction metric
        scores = self._seq(key_character_seq, response_character_seq, pos_class=1)
        recall = scores[0]/(scores[1] + 1e-23)
        precision = scores[2]/(scores[3] + 1e-23)
        movie_coref_metric.character = data.Metric(recall, precision)

        if not only_lea:
            # Get CONLL filenames
            key_word_file = tempfile.NamedTemporaryFile(delete=False).name if output_dir is None else (
                                os.path.join(output_dir, f"gold_epoch.word.{filename}.conll"))
            key_span_file = tempfile.NamedTemporaryFile(delete=False).name if output_dir is None else (
                                os.path.join(output_dir, f"gold_epoch.span.{filename}.conll"))
            response_word_file = tempfile.NamedTemporaryFile(delete=False).name if output_dir is None else (
                                os.path.join(output_dir, f"pred_epoch.word.{filename}.conll"))
            response_span_file = tempfile.NamedTemporaryFile(delete=False).name if output_dir is None else (
                                os.path.join(output_dir, f"pred_epoch.span.{filename}.conll"))
            
            # Convert clusters to CONLL format
            key_word_conll_lines, key_span_conll_lines = [], []
            response_word_conll_lines, response_span_conll_lines = [], []
            for document in documents:
                key_word_conll_lines += conll.convert_to_conll(document, movie_to_key_word_clusters[document.movie])
                key_span_conll_lines += conll.convert_to_conll(document, movie_to_key_span_clusters[document.movie])
                response_word_conll_lines += conll.convert_to_conll(document,
                                                                    movie_to_response_word_clusters[document.movie])
                response_span_conll_lines += conll.convert_to_conll(document,
                                                                    movie_to_response_span_clusters[document.movie])
            
            # Write CONLL files
            with open(key_word_file, "w") as fw:
                fw.writelines(key_word_conll_lines)
            with open(key_span_file, "w") as fw:
                fw.writelines(key_span_conll_lines)
            with open(response_word_file, "w") as fw:
                fw.writelines(response_word_conll_lines)
            with open(response_span_file, "w") as fw:
                fw.writelines(response_span_conll_lines)

            # calculate CONLL metrics
            if scorer_path is None:
                word_metrics = self.pycoref(movie_to_key_word_clusters, movie_to_response_word_clusters,
                                            metrics=["muc", "bcub", "ceafe"])["micro"]
                span_metrics = self.pycoref(movie_to_key_span_clusters, movie_to_response_span_clusters,
                                            metrics=["muc", "bcub", "ceafe"])["micro"]
            else:
                word_metrics = self.perlcoref(key_word_file, response_word_file, scorer_path)["micro"]
                span_metrics = self.perlcoref(key_span_file, response_span_file, scorer_path)["micro"]

            movie_coref_metric.word_coref.muc = word_metrics["muc"]
            movie_coref_metric.word_coref.bcub = word_metrics["bcub"]
            movie_coref_metric.word_coref.ceafe = word_metrics["ceafe"]
            movie_coref_metric.span_coref.muc = span_metrics["muc"]
            movie_coref_metric.span_coref.bcub = span_metrics["bcub"]
            movie_coref_metric.span_coref.ceafe = span_metrics["ceafe"]

            # delete temporary files
            if output_dir is None:
                os.remove(key_word_file)
                os.remove(key_span_file)
                os.remove(response_word_file)
                os.remove(response_span_file)
        
        return movie_coref_metric