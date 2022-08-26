"""Functions to perform data transformations, loading, and saving
"""

from mica_text_coref.coref.seq_coref import data
from mica_text_coref.coref.seq_coref import mention_tree

def load_data(jsonlines_file: str, keep_singletons: bool = False) -> \
    data.CorefCorpus:
    """Load CorefCorpus from jsonlines file and remove overlapping mentions
    from coreference clusters. If the cluster becomes a singleton after
    removing the overlapping mentions, remove or keep it according to the
    keep_singletons argument.
    """
    corpus = data.CorefCorpus(jsonlines_file)
    for document in corpus.documents:
        clusters = []
        for cluster in document.clusters:
            forest = mention_tree.create_mention_forest(cluster)
            mentions: set[data.Mention] = set()
            for mention_node in forest:
                for node in (mention_tree
                .find_largest_subset_of_non_overlapping_mention_nodes(
                    mention_node)):
                    mentions.add(node.mention)
            if keep_singletons or len(mentions) > 1:
                clusters.append(mentions)
        document.clusters = clusters
    return corpus