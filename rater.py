"""Main function to validate raters' annotations against reference"""
from movie_coref import evaluate

from absl import flags
from absl import app
import copy
import os
import pandas as pd
from scorch import scores
import statistics

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", default="data", help="Data directory containing the reference annotations directory.")
evaluator = evaluate.Evaluator()

def validate(raters_csv_files: list[str], reference_csv_file: str):
    """Calculate conll-F1 scores between rater and reference annotations."""
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
    lea_f1s = []
    muc_f1s = []
    for name, clusters in name_and_clusters:
        muc_f1 = scores.muc(reference_clusters, clusters)[0]
        conll_f1 = scores.conll2012(reference_clusters, clusters)
        rN, rD, pN, pD = evaluator._lea(reference_clusters, clusters)
        recall, precision = rN/rD, pN/pD
        lea_f1 = statistics.harmonic_mean([recall, precision])
        conll_f1s.append(conll_f1)
        lea_f1s.append(lea_f1)
        muc_f1s.append(muc_f1)
        print(f"rater {name:20s}: conll-F1 = {conll_f1:.4f} lea-F1 = {lea_f1:.4f} muc-F1 = {muc_f1:.4f}")
    average_conll_f1 = sum(conll_f1s)/len(conll_f1s)
    print(f"average conll-F1 = {average_conll_f1:.4f}")
    average_muc_f1 = sum(muc_f1s)/len(muc_f1s)
    print(f"average muc-F1 = {average_muc_f1:.4f}")
    average_lea_f1 = sum(lea_f1s)/len(lea_f1s)
    print(f"average lea-F1 = {average_lea_f1:.4f}")

def main(_):
    data_dir = FLAGS.data_dir
    raters_csv_files = [os.path.join(data_dir, f"validation/{name}.csv") for name in ["athashree", "chakor", "prithvi"]]
    reference_csv_file = os.path.join(data_dir, "validation/reference.csv")
    validate(raters_csv_files, reference_csv_file)

if __name__=="__main__":
    app.run(main)