# author: Sabyasachee

# third party
import pandas as pd

# custom
from moviecoref.scores import conll2012

def interrater_agreement():
    athashree = pd.read_csv("data/annotation/validation/athashree.csv")
    chakor = pd.read_csv("data/annotation/validation/chakor.csv")
    prithvi = pd.read_csv("data/annotation/validation/prithvi.csv")
    reference = pd.read_csv("data/annotation/validation/reference.csv")

    athashree_clusters = []
    chakor_clusters = []
    prithvi_clusters = []
    reference_clusters = []

    for _, character_df in athashree.groupby("entityLabel"):
        cluster = set()
        for _, row in character_df.iterrows():
            begin, end = int(row["begin"]), int(row["end"])
            cluster.add((begin, end))
        athashree_clusters.append(cluster)

    for _, character_df in chakor.groupby("entityLabel"):
        cluster = set()
        for _, row in character_df.iterrows():
            begin, end = int(row["begin"]), int(row["end"])
            cluster.add((begin, end))
        chakor_clusters.append(cluster)

    for _, character_df in prithvi.groupby("entityLabel"):
        cluster = set()
        for _, row in character_df.iterrows():
            begin, end = int(row["begin"]), int(row["end"])
            cluster.add((begin, end))
        prithvi_clusters.append(cluster)

    for _, character_df in reference.groupby("entityLabel"):
        cluster = set()
        for _, row in character_df.iterrows():
            begin, end = int(row["begin"]), int(row["end"])
            cluster.add((begin, end))
        reference_clusters.append(cluster)

    athashree_score = conll2012(reference_clusters, athashree_clusters)
    chakor_score = conll2012(reference_clusters, chakor_clusters)
    prithvi_score = conll2012(reference_clusters, prithvi_clusters)

    print("conll-2012 coreference score: athashree = {:.3f}, chakor = {:.3f}, prithvi = {:.3f}".format(athashree_score, chakor_score, prithvi_score))
    print("average conll-2012 coreference score = {:.3f}".format((athashree_score + chakor_score + prithvi_score)/3))