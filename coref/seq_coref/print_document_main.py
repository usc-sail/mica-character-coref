"""Pretty-print ontonotes coreference document

Given a document_key, pretty print the document to stdout.
The output is formatted as follows:
    <Speaker> : <sentence>
    <Speaker> : <sentence>

    <Cluster> : <Mentions>
    <Cluster> : <Mentions>

    Usage:
    
    python print_document --doc_key=<doc_key>
"""

from mica_text_coref.coref.seq_coref import data
from mica_text_coref.coref.seq_coref import util

from absl import flags
from absl import app
import jsonlines

FLAGS = flags.FLAGS
flags.DEFINE_string("conll_jsonlines", None,
                    "Path to conll-2012 gold jsonlines file", required=True)
flags.DEFINE_string("doc_key", None,
                    "conll-2012 document key e.g. bc/msnbc/00/msnbc_0000_12", required=True)

def pretty_print_document():
    with jsonlines.open(FLAGS.conll_jsonlines) as reader:
        for json in reader:
            if json["doc_key"] == FLAGS.doc_key:
                coref_document = data.CorefDocument(json)
                print(util.pretty_format_coref_document(coref_document))
                break
        else:
            print(f"doc_key={FLAGS.doc_key} not found in jsonlines file={FLAGS.conll_jsonlines}")

def main(argv):
    pretty_print_document()

if __name__=="__main__":
    app.run(main)