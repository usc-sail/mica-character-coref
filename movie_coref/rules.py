"""Coreference resolution rules"""
# pyright: reportGeneralTypeIssues=false

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

def _keep(flags: list[bool], clusters: list[set[tuple[int, int]]], conjunction = True) -> list[set[tuple[int, int]]]:
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

def keep_speakers(parse_tags: list[str], clusters: list[set[tuple[int, int]]]) -> list[set[tuple[int, int]]]:
    """Retain speaker clusters."""
    is_speaker = [tag == "C" for tag in parse_tags]
    return _keep(is_speaker, clusters)

def keep_persons(ner_tags: list[str], clusters: list[set[tuple[int, int]]]) -> list[set[tuple[int, int]]]:
    """Retain person clusters."""
    is_person = [tag == "PERSON" for tag in ner_tags]
    return _keep(is_person, clusters)

def merge_speakers(words: list[str], parse_tags: list[str], clusters: list[set[tuple[int, int]]]) -> (
        list[set[tuple[int, int]]]):
    """Merge clusters that contain speaker mentions with the same name."""
    cluster_nodes = [ClusterNode(i) for i in range(len(clusters))]
    parse_tags = np.array(parse_tags) # type: ignore
    cluster_speakers = [set([re.sub(r"\([^\)]+\)", "", " ".join(words[i: j + 1])).upper().strip()
                             for i, j in cluster if all(parse_tags[i: j + 1] == "C")]) # type: ignore
                             for cluster in clusters]

    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            if not cluster_speakers[i].isdisjoint(cluster_speakers[j]):
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

def filter_mentions(mentions: set[tuple[int, int]], clusters: list[set[tuple[int, int]]]) -> list[set[tuple[int, int]]]:
    """Filter mentions in clusters by the given mentions list."""
    _clusters = []
    for cluster in clusters:
        cluster = cluster.intersection(mentions)
        if len(cluster) > 0:
            _clusters.append(cluster)
    return _clusters

def remove_singleton_clusters(clusters: list[set[tuple[int, int]]]) -> list[set[tuple[int, int]]]:
    """Remove clusters containing one mention."""
    return list(filter(lambda cluster: len(cluster) > 1, clusters))

def remove_speaker_links(clusters: list[tuple[int, int]], parse: list[str]) -> list[tuple[int, int]]:
    """Remove mentions from cluster if it contains a word with speaker parse tag.

    Args:
        clusters (list[set]): List of clusters.
        parse (list): List of word-level parse tags.
    
    Return:
        List of clusters.
    """
    clusters_ = []
    for cluster in clusters:
        cluster_ = set()
        for begin, end in cluster:
            if np.all(parse[begin: end + 1] != "C"):
                cluster_.add((begin, end))
        if cluster_:
            clusters_.append(cluster_)
    return clusters_