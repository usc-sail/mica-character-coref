import scorch.scores

def evaluate_coreference(gold_clusters, sys_clusters, verbose=True):
    muc_recall, muc_precision, muc_F1 = scorch.scores.muc(gold_clusters, sys_clusters)
    bcubed_recall, bcubed_precision, bcubed_F1 = scorch.scores.b_cubed(gold_clusters, sys_clusters)
    ceafe_recall, ceafe_precision, ceafe_F1 = scorch.scores.ceaf_e(gold_clusters, sys_clusters)
    conll_recall = (muc_recall + bcubed_recall + ceafe_recall)/3
    conll_precision = (muc_precision + bcubed_precision + ceafe_precision)/3
    conll_F1 = (muc_F1 + bcubed_F1 + ceafe_F1)/3
    
    if verbose:
        print(f"MUC  : P = {muc_precision:.4f} R = {muc_recall:.4f} F1 = {muc_F1:.4f}")
        print(f"B3   : P = {bcubed_precision:.4f} R = {bcubed_recall:.4f} F1 = {bcubed_F1:.4f}")
        print(f"CEAFe: P = {ceafe_precision:.4f} R = {ceafe_recall:.4f} F1 = {ceafe_F1:.4f}")
        print(f"CoNLL 2012 score: {conll_F1:.4f}")

    result = {  "muc":      {"R": muc_recall,       "P": muc_precision,     "F1": muc_F1}, 
                "bcubed":   {"R": bcubed_recall,    "P": bcubed_precision,  "F1": bcubed_F1}, 
                "ceafe":    {"R": ceafe_recall,     "P": ceafe_precision,   "F1": ceafe_F1},
                "conll2012":{"R": conll_recall,     "P": conll_precision,   "F1": conll_F1}}
    return result