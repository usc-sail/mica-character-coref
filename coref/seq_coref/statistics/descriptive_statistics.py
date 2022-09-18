"""Functions to compute descriptive statistics of coreference documents.
"""

from mica_text_coref.coref.seq_coref.data import data
from mica_text_coref.coref.seq_coref.data import mention_tree
from mica_text_coref.coref.seq_coref.utils import util

import collections
import numpy as np

def find_mention_pair_relationships_in_cluster(
    cluster: set[data.Mention]) -> dict[data.MentionPairRelationship, int]:
    """Find distribution of MentionPairRelationship types in cluster. If the
    cluster has n mentions, then there will be nC2 = n(n-1)/2 relationships.

    Args:
        cluster: Set of data.Mention objects. It represents a coreference chain.
    
    Returns:
        A dictionary of data.MentionPairRelationship to count, representing
        mention overlap distribution.
    """
    mention_pair_relationships: list[data.MentionPairRelationship] = []
    cluster = list(cluster)
    for i in range(len(cluster)):
        for j in range(i + 1, len(cluster)):
            mention_pair_relationship = util.find_mention_pair_relationship(
                cluster[i], cluster[j])
            mention_pair_relationships.append(mention_pair_relationship)
    distribution = dict(collections.Counter(mention_pair_relationships))
    return distribution

def find_maximum_number_of_mentions_covering_a_single_word(
    cluster: set[data.Mention]) -> int:
    """Find the maximum number of mentions that overlap on a single word.

    Args:
        cluster: Set of data.Mention objects, representing a coreference chain.
    
    Returns:
        The maximum number of mentions that overlap on some word.
    """
    end_index = max([mention.end for mention in cluster])
    n_mentions_covering_word_index = np.zeros(end_index + 1, dtype=int)
    for mention in cluster:
        n_mentions_covering_word_index[mention.begin: mention.end + 1] += 1
    return n_mentions_covering_word_index.max()

def find_largest_subset_of_non_overlapping_mentions(
    cluster: set[data.Mention]) -> set[data.Mention]:
    """Find the largest subset of non-overlapping mentions in cluster.

    Args:
        cluster: Set of data.Mention objects, representing a coreference chain.
    
    Returns:
        A set of non-overlapping data.Mention objects which is the largest
        subset of the given cluster.
    """
    forest = mention_tree.create_mention_forest(cluster)
    mentions: set[data.Mention] = set()
    for mention_node in forest:
        for node in (mention_tree
        .find_largest_subset_of_non_overlapping_mention_nodes(
            mention_node)):
            mentions.add(node.mention)
    return mentions