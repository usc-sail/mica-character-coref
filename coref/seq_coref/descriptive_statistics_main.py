"""Find descriptive statistics for each (language, partition) dataset.

The descriptive statistics we find are:
    1. Total number of mentions
    2. Total number of entities
    3. MentionPairRelationship distribution
    4. Total number of mentions and entities when overlaps are removed from an entity
"""

from mica_text_coref.coref.seq_coref import data
from mica_text_coref.coref.seq_coref import descriptive_statistics

from absl import flags
from absl import app
import os

FLAGS = flags.FLAGS
flags.DEFINE_string("conll_directory", None, "Directory containing conll gold jsonlines files",
                    required=True)

def find_descriptive_statistics(language:str, partition:str):
    """Find descriptive statistics of given language, partition dataset"""
    jsonfiles_path = os.path.join(FLAGS.conll_directory, f"{partition}.{language}.jsonlines")
    coref_corpus = data.CorefCorpus(jsonfiles_path)

    total_n_mentions = 0
    total_n_entities = 0
    total_n_mentions_no_overlap = 0
    total_n_mentions_no_overlap_without_singleton = 0
    total_n_entities_no_overlap_without_singleton = 0
    mention_pair_relationship_distribution: dict[data.MentionPairRelationship, int] = {}
    maximum_number_of_mentions_covering_some_word = 0

    for document in coref_corpus.documents:
        for cluster in document.clusters:
            total_n_entities += 1
            total_n_mentions += len(cluster)
            
            distribution = descriptive_statistics.find_mention_pair_relationships_in_cluster(
                cluster)
            for relationship_type, count in distribution.items():
                if relationship_type not in mention_pair_relationship_distribution:
                    mention_pair_relationship_distribution[relationship_type] = 0
                mention_pair_relationship_distribution[relationship_type] += count
            
            max_number = (
                descriptive_statistics.find_maximum_number_of_mentions_covering_a_single_word(
                    cluster))
            maximum_number_of_mentions_covering_some_word = max(
                maximum_number_of_mentions_covering_some_word, max_number)

            non_overlapping_cluster = (
                descriptive_statistics.find_largest_subset_of_non_overlapping_mentions(cluster))
            total_n_mentions_no_overlap += len(non_overlapping_cluster)
            if len(non_overlapping_cluster) > 1:
                total_n_mentions_no_overlap_without_singleton += len(non_overlapping_cluster)
                total_n_entities_no_overlap_without_singleton += 1
            
    print(f"{language}.{partition}")
    print(f"Total mentions = {total_n_mentions}, total entities = {total_n_entities}")
    
    print(f"Mention pair relationship distribution:")
    total_n_mention_pairs = sum(mention_pair_relationship_distribution.values())
    for relationship_type, count in mention_pair_relationship_distribution.items():
        percentage = 100*(count/total_n_mention_pairs)
        print(f"\t{relationship_type}: {count} ({percentage:.2f}%)")
    
    print(f"Maximum number of mentions covering a token = "
          f"{maximum_number_of_mentions_covering_some_word}")
    
    mention_percentage_decrease = 100*(
        total_n_mentions - total_n_mentions_no_overlap)/total_n_mentions
    mention_percentage_decrease_without_singleton = 100*(
        total_n_mentions - total_n_mentions_no_overlap_without_singleton)/total_n_mentions
    entity_percentage_decrease_without_singleton = 100*(
        total_n_entities - total_n_entities_no_overlap_without_singleton)/total_n_entities
    print("Filter mentions from cluster s.t. the largest non-overlapping subset is kept")
    print(f"\tTotal mentions = {total_n_mentions_no_overlap} (-{mention_percentage_decrease:.2f}%),"
          f" total entities = {total_n_entities}")
    print("Filter (as above) + remove singleton entities")
    print(f"\tTotal mentions = {total_n_mentions_no_overlap_without_singleton} "
          f"(-{mention_percentage_decrease_without_singleton:.2f}%),"
          f" total entities = {total_n_entities_no_overlap_without_singleton} "
          f"(-{entity_percentage_decrease_without_singleton:.2f}%)")
    print()

def main(argv):
    for language in ["english", "chinese", "arabic"]:
        for partition in ["train", "dev", "test"]:
            find_descriptive_statistics(language, partition)

if __name__=="__main__":
    app.run(main)