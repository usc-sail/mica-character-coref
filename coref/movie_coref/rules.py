"""Coreference resolution rules"""

import numpy as np
import re

class ClusterNode:
    """Graph of clusters with edges connecting co-referring clusters."""
    def __init__(self, cluster_id: int):
        self.id = cluster_id
        self.neighbors: set[ClusterNode] = set()
        self.visited = False

    def link(self, other: "ClusterNode"):
        self.neighbors.add(other)
        other.neighbors.add(self)

def _keep(flags: list[bool], clusters: list[set[tuple[int, int]]], conjunction = True) -> (
    list[set[tuple[int, int]]]):
    """Retain cluster if atleast one of its mention has all True word flags. If conjunction is
    False, mention should have atleast one True word flag.

    Args:
        flags: List of word-level boolean flags.
        clusters: List of clusters. Each cluster is a set of mentions. Each mention is a set of
            2-element integer tuples.

    Returns:
        Filtered list of clusters.
    """
    speaker_clusters = []
    for cluster in clusters:
        contains_speaker = False
        for i, j in cluster:
            if conjunction:
                mention_flag = all(flags[i: j + 1])
            else:
                mention_flag = any(flags[i: j + 1])
            if mention_flag:
                contains_speaker = True
                break
        if contains_speaker:
            speaker_clusters.append(cluster)
    return speaker_clusters

def keep_speakers(parse_tags: list[str], clusters: list[set[tuple[int, int]]]) -> (
    list[set[tuple[int, int]]]):
    """Retain speaker clusters."""
    is_speaker = [tag == "C" for tag in parse_tags]
    return _keep(is_speaker, clusters)

def keep_persons(ner_tags: list[str], clusters: list[set[tuple[int, int]]]) -> (
    list[set[tuple[int, int]]]):
    """Retain person clusters."""
    is_person = [tag == "PERSON" for tag in ner_tags]
    return _keep(is_person, clusters)

def _merge(words: list[str], parse_tags: list[str], cluster_x: set[tuple[int, int]], 
    cluster_y: set[tuple[int, int]]):
    """Returns true if cluster have speakers with same names."""
    speakers_x = set([re.sub(r"\([^\)]+\)", "", " ".join(words[i: j + 1])).upper().strip()
        for i, j in cluster_x if all(parse_tags[i: j + 1]) == "C"])
    speakers_y = set([re.sub(r"\([^\)]+\)", "", " ".join(words[i: j + 1])).upper().strip()
        for i, j in cluster_y if all(parse_tags[i: j + 1]) == "C"])
    return not speakers_x.isdisjoint(speakers_y)

def merge_speakers(words: list[str], parse_tags: list[str], clusters: list[set[tuple[int, int]]]
    ) -> list[set[tuple[int, int]]]:
    """Merge clusters that contain speaker mentions with the same name."""
    cluster_nodes = [ClusterNode(i) for i in range(len(clusters))]

    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            if _merge(words, parse_tags, clusters[i], clusters[j]):
                cluster_nodes[i].link(cluster_nodes[j])

    _clusters = []
    for node in cluster_nodes:
        if not node.visited:
            cluster_ids = set([])
            stack = [node]
            while stack:
                current_node = stack.pop()
                current_node.visited = True
                cluster_ids.add(current_node.id)
                stack.extend(_node for _node in current_node.neighbors if not _node.visited)
            cluster = set([])
            for _id in cluster_ids:
                cluster.update(clusters[_id])
            _clusters.append(cluster)

    return _clusters