"""Split screenplays of the training set for training with word level coreference model"""

from mica_text_coref.coref.movie_coref import split_and_merge

from absl import app
from absl import flags
import jsonlines
import os

FLAGS = flags.FLAGS
data_dir = os.getenv("DATA_DIR")

flags.DEFINE_string("data_dir", default=os.path.join(data_dir, "mica_text_coref/movie_coref/results"), help="directory containing directories that contain processed screenplay jsonlines files")
flags.DEFINE_list("data_subdir", default=["regular", "nocharacters", "addsays"], help="directory names inside data_dir that contain processed screenplay jsonlines files which needs to be split")
flags.DEFINE_list("split_len", default=[1024, 2048, 3072, 4096, 5120], help="length of the smaller screenplay in words")

def split_screenplays_for_wl(argv):
    # check argv
    if len(argv) > 1:
        print("too many arguments")
        return
    
    # split screenplay
    partitions = ["train", "dev"]
    for subdir in FLAGS.data_subdir:
        for partition in partitions:
            for split_len in FLAGS.split_len:
                print(f"subdir = {subdir}, partition = {partition}, split_len = {split_len}")
                movie_file = os.path.join(FLAGS.data_dir, subdir, f"{partition}_wl.jsonlines")
                output_file = os.path.join(FLAGS.data_dir, subdir, f"{partition}_wl.split_{split_len}.jsonlines")
                with jsonlines.open(movie_file, mode="r") as reader, jsonlines.open(output_file, mode="w") as writer:
                    for doc in reader:
                        for subdoc in split_and_merge.split_screenplay(doc, split_len, 0, verbose=True):
                            if len(subdoc["clusters"]) > 0:
                                head2span: set[tuple[int, int, int]] = set()
                                word_clusters: list[set[int]] = []
                                span_clusters: list[set[tuple[int, int]]] = []
                                for _, cluster in subdoc["clusters"].items():
                                    word_cluster: set[int] = set()
                                    span_cluster: set[tuple[int, int]] = set()
                                    for begin, end, head in cluster:
                                        head2span.add((begin, end + 1, head))
                                        word_cluster.add(head)
                                        span_cluster.add((begin, end + 1))
                                    word_clusters.append(word_cluster)
                                    span_clusters.append(span_cluster)
                                subdoc["part_id"] = 0
                                subdoc["head2span"] = [[head, begin, end] for head, begin, end in head2span]
                                subdoc["word_clusters"] = [sorted(word_cluster) for word_cluster in word_clusters]
                                subdoc["span_clusters"] = [sorted([list(span) for span in span_cluster]) for span_cluster in span_clusters]
                                writer.write(subdoc)

if __name__=="__main__":
    app.run(split_screenplays_for_wl)