"""Function to pretty format a coreference document"""

from mica_text_coref.coref.seq_coref import data
from mica_text_coref.coref.seq_coref import util

def pretty_format_coref_document(document: data.CorefDocument) -> str:
    """Pretty-format coreference document"""
    desc = "DOC_KEY\n"
    desc += "=======\n"
    desc += f"{document.doc_key}\n\n"
    desc += "DOCUMENT\n"
    desc += "========\n"
    document_words = [word for sentence in document.sentences for word in sentence]

    for sentence_speakers, sentence_words in zip(document.speakers, document.sentences):
        speaker = sentence_speakers[0]
        block = util.indent_block(sentence_words, 20, 80)
        desc += f"{speaker:17s} : " + block + "\n"

    desc += "\n"
    desc += "CLUSTERS\n"
    desc += "========\n"

    for i, cluster in enumerate(document.clusters):
        cluster_index = f"Cluster {i + 1}"
        mentions = []
        for mention in cluster:
            mention = "[" + " ".join(document_words[mention.begin: mention.end + 1]) + "]"
            mentions.append(mention)
        block = util.indent_block(mentions, 20, 80)
        desc += f"{cluster_index:17s} : " + block + "\n"
    return desc