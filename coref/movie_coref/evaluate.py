"""Evaluate coreference"""

import math
import typing
import numpy as np
from scipy.optimize import linear_sum_assignment

def trace(cluster: set, partition: list[set]):
    remaining = set(cluster)
    for a in partition:
        common = remaining.intersection(a)
        if common:
            remaining.difference_update(common)
            yield common
    for x in sorted(remaining):
        yield set((x,))

def muc(key: list[set], response: list[set]) -> tuple[float, float, float, float]:
    if all(len(k) == 1 for k in key) or all(len(r) == 1 for r in response):
        return 0.0, 0.0, 0.0, 0.0
    R_numer = sum(len(k) - sum(1 for _ in trace(k, response)) for k in key)
    R_denom = sum(len(k) - 1 for k in key)
    P_numer = sum(len(r) - sum(1 for _ in trace(r, key)) for r in response)
    P_denom = sum(len(r) - 1 for r in response)
    return R_numer, R_denom, P_numer, P_denom

def b_cubed(key: list[set], response: list[set]) -> tuple[float, float, float, float]:
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

def ceaf(key: list[set], response: list[set], score: typing.Callable[[set, set], float]) -> tuple[float, float, float, float]:
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

def ceaf_m(key: list[set], response: list[set]) -> tuple[float, float, float, float]:
    def Φ_3(k, r):
        return len(k.intersection(r))
    return ceaf(key, response, Φ_3)

def ceaf_e(key: list[set], response: list[set]) -> tuple[float, float, float, float]:
    def Φ_4(k, r):
        return 2 * len(k.intersection(r)) / (len(k) + len(r))
    return ceaf(key, response, Φ_4)

def find_prf_from_nds(r_numer: float, r_denom: float, p_numer: float, p_denom: float) -> tuple[float, float, float]:
    r = 100 * r_numer / (1e-23 + r_denom)
    p = 100 * p_numer / (1e-23 + p_denom)
    f1 = 2 * p * r / (1e-23 + p + r)
    return r, p, f1

def evaluate(movie_to_gold_clusters: dict[str, list[set[tuple[int, int]]]], movie_to_pred_clusters: dict[str, list[set[tuple[int, int]]]]) -> dict[str, dict[str, tuple[float, float, float]]]:
    assert set(movie_to_gold_clusters.keys()) == set(movie_to_pred_clusters.keys())
    all_muc_R_numer, all_muc_R_denom, all_muc_P_numer, all_muc_P_denom = 0, 0, 0, 0
    all_b_cubed_R_numer, all_b_cubed_R_denom, all_b_cubed_P_numer, all_b_cubed_P_denom = 0, 0, 0, 0
    all_ceaf_e_R_numer, all_ceaf_e_R_denom, all_ceaf_e_P_numer, all_ceaf_e_P_denom = 0, 0, 0, 0
    result: dict[str, dict[str, tuple[float, float, float]]] = {}
    for movie in movie_to_pred_clusters.keys():
        gold_clusters, pred_clusters = movie_to_gold_clusters[movie], movie_to_pred_clusters[movie]
        muc_R_numer, muc_R_denom, muc_P_numer, muc_P_denom = muc(gold_clusters, pred_clusters)
        b_cubed_R_numer, b_cubed_R_denom, b_cubed_P_numer, b_cubed_P_denom = b_cubed(gold_clusters, pred_clusters)
        ceaf_e_R_numer, ceaf_e_R_denom, ceaf_e_P_numer, ceaf_e_P_denom = ceaf_e(gold_clusters, pred_clusters)
        all_muc_R_numer += muc_R_numer; all_muc_R_denom += muc_R_denom; all_muc_P_numer += muc_P_numer; all_muc_P_denom += muc_P_denom
        all_b_cubed_R_numer += b_cubed_R_numer; all_b_cubed_R_denom += b_cubed_R_denom; all_b_cubed_P_numer += b_cubed_P_numer; all_b_cubed_P_denom += b_cubed_P_denom
        all_ceaf_e_R_numer += ceaf_e_R_numer; all_ceaf_e_R_denom += ceaf_e_R_denom; all_ceaf_e_P_numer += ceaf_e_P_numer; all_ceaf_e_P_denom += ceaf_e_P_denom
        result[movie] = {}
        result[movie]["muc"] = find_prf_from_nds(muc_R_numer, muc_R_denom, muc_P_numer, muc_P_denom)
        result[movie]["bcub"] = find_prf_from_nds(b_cubed_R_numer, b_cubed_R_denom, b_cubed_P_numer, b_cubed_P_denom)
        result[movie]["ceafe"] = find_prf_from_nds(ceaf_e_R_numer, ceaf_e_R_denom, ceaf_e_P_numer, ceaf_e_P_denom)
    result["all"] = {}
    result["all"]["muc"] = find_prf_from_nds(all_muc_R_numer, all_muc_R_denom, all_muc_P_numer, all_muc_P_denom)
    result["all"]["bcub"] = find_prf_from_nds(all_b_cubed_R_numer, all_b_cubed_R_denom, all_b_cubed_P_numer, all_b_cubed_P_denom)
    result["all"]["ceafe"] = find_prf_from_nds(all_ceaf_e_R_numer, all_ceaf_e_R_denom, all_ceaf_e_P_numer, all_ceaf_e_P_denom)
    return result