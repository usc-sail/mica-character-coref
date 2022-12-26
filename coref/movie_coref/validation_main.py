"""Main function to validate raters' annotations against reference"""

from absl import flags
from absl import app
import copy
import os
import pandas as pd
from scorch import scores

FLAGS = flags.FLAGS
proj_dir = os.getcwd()
flags.DEFINE_multi_string(
    "rater",
    default=[os.path.join(proj_dir, f"data/movie_coref/validation/{name}.csv")
                for name in ["athashree", "chakor", "prithvi"]],
    help="Rater csv annotation file(s).")
flags.DEFINE_string(
    "reference",
    default=os.path.join(proj_dir, "data/movie_coref/validation/reference.csv"),
    help="Reference csv annotation file.")

def validate(raters_csv_files: list[str], reference_csv_file: str):
    """Calculate conll-F1 scores between rater and reference annotations.
    """
    name_and_clusters = []
    reference_clusters = None

    for csv_file in raters_csv_files + [reference_csv_file]:
        df = pd.read_csv(csv_file)
        clusters = []
        for _, character_df in df.groupby("entityLabel"):
            cluster = set()
            for _, row in character_df.iterrows():
                begin, end = int(row["begin"]), int(row["end"])
                cluster.add((begin, end))
            clusters.append(cluster)
        if csv_file != reference_csv_file:
            name = os.path.basename(csv_file).rstrip(".csv")
            name_and_clusters.append((name, clusters))
        else:
            reference_clusters = copy.deepcopy(clusters)
    
    conll_f1s = []
    for name, clusters in name_and_clusters:
        conll_f1 = scores.conll2012(reference_clusters, clusters)
        conll_f1s.append(conll_f1)
        print(f"rater {name:20s}: conll-F1 against reference = {conll_f1:.2f}")
    average_conll_f1 = sum(conll_f1s)/len(conll_f1s)
    print(f"average conll-F1 = {average_conll_f1:.2f}")

def main(argv):
    validate(FLAGS.rater, FLAGS.reference)

if __name__=="__main__":
    app.run(main)