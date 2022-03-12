def evaluate_mention_identification(gold_clusters, sys_clusters):
    gold_mentions = set([mention for cluster in gold_clusters for mention in cluster])
    sys_mentions = set([mention for cluster in sys_clusters for mention in cluster])
    u, v, w = len(gold_mentions.intersection(sys_mentions)), len(gold_mentions), len(sys_mentions)
    p, r, f1 = u/w, u/v, 2*u/(v + w)
    return {"mention_P": p, "mention_R": r, "mention_F1": f1}