"""Functions to compute forest of mentions from document to find the maximum
non overlapping sequence.
"""

from mica_text_coref.coref.seq_coref import data
from mica_text_coref.coref.seq_coref import util

class MentionNode:
    """Data structure to represent a mention in a tree/forest of mentions.
    """

    def __init__(self, mention:data.Mention) -> None:
        self.mention = mention
        self.contained: list[MentionNode] = []
        self.value = 1
    
    def add_contained_node(self, mention_node:"MentionNode"):
        self.contained.append(mention_node)
        contained_value = sum(node.value for node in self.contained)
        self.value = max(1, contained_value)
    
def create_mention_forest(cluster: set[data.Mention]) -> list[MentionNode]:
    """Create a forest of mention trees from cluster of mentions. An edge
    between a parent mention_tree.MentionNode and child
    mention_tree.MentionNode in a tree represents that the mention of the
    parent node subsumes the mention in the child node. Mentions in different
    trees are disjoint from each other.

    Args:
        cluster: Set of data.Mention objects, representing a coreference chain.
    
    Returns:
        A list of mention_tree.MentionNode, each being a root of a tree.
    """
    sorted_cluster = sorted(cluster, key=lambda mention: (mention.begin,
    -len(mention)))
    i = 0
    mention_forest = []

    while i < len(sorted_cluster):
        j = i + 1
        contained_cluster: set[data.Mention] = set()
        while j < len(sorted_cluster) and util.find_mention_pair_relationship(
            sorted_cluster[i], sorted_cluster[j]) == (
                data.MentionPairRelationship.SUBSPAN):
            contained_cluster.add(sorted_cluster[j])
            j = j + 1
        mention_node = MentionNode(sorted_cluster[i])
        if contained_cluster:
            contained_mention_nodes = create_mention_forest(contained_cluster)
            for node in contained_mention_nodes:
                mention_node.add_contained_node(node)
        mention_forest.append(mention_node)
        i = j
    
    return mention_forest

def find_largest_subset_of_non_overlapping_mention_nodes(
    mention_node: MentionNode) -> list[MentionNode]:
    """Find the largest set of non-overlapping mention nodes in the tree rooted
    at given mention node.
    
    Args:
        mention_node: A mention_tree.MentionNode which is the root of a tree.
    
    Returns:
        A list of mention_tree.MentionNode which does not overlap with each
        other and is the largest subset of the tree.
    """
    if mention_node.value == 1:
        return [mention_node]
    else:
        non_overlapping_nodes = []
        for contained_node in mention_node.contained:
            for node in find_largest_subset_of_non_overlapping_mention_nodes(
                contained_node):
                non_overlapping_nodes.append(node)
        return non_overlapping_nodes