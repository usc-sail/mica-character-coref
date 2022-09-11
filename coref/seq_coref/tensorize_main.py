"""Convert english coreference corpus to tensors for training and testing using
longformer sequence coreference model, and save the tensors.
"""

from mica_text_coref.coref.seq_coref import data
from mica_text_coref.coref.seq_coref import data_util
from mica_text_coref.coref.seq_coref import tensorize
from mica_text_coref.coref.seq_coref import representatives

from absl import flags
from absl import app
import os
import torch
from transformers import LongformerTokenizer

FLAGS = flags.FLAGS
flags.DEFINE_string("conll_directory", None,
                    "Directory containing English conll gold jsonlines files",
                    required=True)
flags.DEFINE_string("longformer_tensors_directory", None,
                    "Directory to which the tensors will be saved",
                    required=True)

def create_and_save_tensors():
    """Create tensors for training and testing the English coreference
    dataset using the longformer sequence coreference model.
    """
    tokenizer: LongformerTokenizer = LongformerTokenizer.from_pretrained(
        "allenai/longformer-base-4096")
    total_n_clusters = 0
    total_n_seq_clusters = 0
    total_n_tensor_seq_clusters = 0

    for partition in ["train", "test", "dev"]:
        print(f"partition = {partition}")
        jsonfiles_path = os.path.join(FLAGS.conll_directory, 
                                      f"{partition}.english.jsonlines")
        corpus = data.CorefCorpus(jsonfiles_path, 
                                  use_ascii_transliteration=True)
        n_clusters = sum(len(document.clusters) for document in 
                            corpus.documents)
        seq_corpus = data_util.remove_overlaps(corpus, keep_singletons=False)
        n_seq_clusters = sum(len(document.clusters) for document in
                                seq_corpus.documents)

        mentions: list[list[data.Mention]] = []
        for document in seq_corpus.documents:
            document_mentions: list[data.Mention] = []
            for cluster in document.clusters:
                mention = representatives.representative_mention(
                    cluster, document)
                document_mentions.append(mention)
            mentions.append(document_mentions)

        longformer_seq_corpus = data_util.remap_spans_document_level(
            seq_corpus, tokenizer.tokenize)
        dataset = tensorize.create_tensors(longformer_seq_corpus, mentions,
                                            tokenizer)
        n_tensor_seq_clusters = dataset.tensors[0].shape[0]

        (token_ids, mention_ids, label_ids, attn_mask, global_attn_mask,
            doc_ids) = dataset.tensors
        directory = os.path.join(FLAGS.longformer_tensors_directory, partition)
        os.makedirs(directory, exist_ok=True)
        print(f"saving {partition} tensors to {directory}")
        torch.save(token_ids, os.path.join(directory, "tokens.pt"))
        torch.save(mention_ids, os.path.join(directory, "mentions.pt"))
        torch.save(label_ids, os.path.join(directory, "labels.pt"))
        torch.save(attn_mask, os.path.join(directory, "attn.pt"))
        torch.save(global_attn_mask, os.path.join(directory, "global_attn.pt"))
        torch.save(doc_ids, os.path.join(directory, "docs.pt"))

        print(f"Number of clusters = {n_clusters}")
        print(f"Number of clusters after removing overlaps = {n_seq_clusters}")
        print(f"Number of clusters after removing overlaps and tensorizing = "
              f"{n_tensor_seq_clusters}\n")

        total_n_clusters += n_clusters
        total_n_seq_clusters += n_seq_clusters
        total_n_tensor_seq_clusters += n_tensor_seq_clusters

    print(f"Number of clusters = {total_n_clusters}")
    print(f"Number of clusters after removing overlaps = "
          f"{total_n_seq_clusters}")
    print(f"Number of clusters after removing overlaps and tensorizing = "
          f"{total_n_tensor_seq_clusters}\n")

def main(argv):
    create_and_save_tensors()

if __name__=="__main__":
    app.run(main)