import argparse
import pandas as pd

def print_evaluation_numbers(results_file):
    results = pd.read_csv(results_file, index_col=None)
    results = results[~results.keep_only_speaker_gold_clusters & ~results.remove_singleton_gold_clusters & ~results.heuristic_pronoun_resolution & (results.min_speaker_sim == 0.9) & (results.max_speaker_merges == 5)]
    conll_cols = ["muc_P", "muc_R", "muc_F1", "bcubed_P", "bcubed_R", "bcubed_F1", "ceafe_P", "ceafe_R", "ceafe_F1", "conll2012_P", "conll2012_R", "conll2012_F1"]
    nec_cols = ["nec_F1", "nec_name_F1", "nec_pronoun_F1", "nec_nominal_F1"]
    mention_cols = ["mention_P", "mention_R", "mention_F1"]
    config_cols = ["use_speaker_sep", "keep_speaker_sys_clusters", "keep_person_sys_clusters", "heuristic_speaker_resolution"]
    hparam_cols = ["min_speaker_sim", "max_speaker_merges"]

    for hparam_col in hparam_cols:
        values = results[hparam_col].unique()
        if len(values) == 1:
            print(f"{hparam_col} = {values[0]}")
        else:
            print(f"{hparam_col} = {values}")
    print()

    baseline = results.loc[~results.use_speaker_sep & results.keep_person_sys_clusters & ~results.keep_speaker_sys_clusters & ~results.heuristic_speaker_resolution, conll_cols + nec_cols + mention_cols].mean()
    print("Baseline")
    pr = []
    for metric in conll_cols + nec_cols + mention_cols:
        x = f"{100*baseline[metric]:.1f}"
        print(f"\t{metric:20s} = {baseline[metric]:.3f}")
        pr.append(x)
    print(" & ".join(pr))
    print()

    model = results.loc[results.use_speaker_sep & ~results.keep_person_sys_clusters & results.keep_speaker_sys_clusters & results.heuristic_speaker_resolution, conll_cols + nec_cols + mention_cols].mean()
    print("Model")
    pr = []
    for metric in conll_cols + nec_cols + mention_cols:
        x = f"{100*model[metric]:.1f}"
        print(f"\t{metric:20s} = {model[metric]:.3f}")
        pr.append(x)
    print(" & ".join(pr))
    print()

    model_minus_speaker_sep = results.loc[~results.use_speaker_sep & ~results.keep_person_sys_clusters & results.keep_speaker_sys_clusters & results.heuristic_speaker_resolution, conll_cols + nec_cols + mention_cols].mean()
    print("Model - Add says")
    pr = []
    for metric in conll_cols + nec_cols + mention_cols:
        x = f"{100*(model_minus_speaker_sep[metric] - model[metric]):.1f}"
        print(f"\t{metric:20s} = {model_minus_speaker_sep[metric] - model[metric]:.3f}")
        pr.append(x)
    print(" & ".join(pr))
    print()

    model_minus_keep_speakers = results.loc[results.use_speaker_sep & results.keep_person_sys_clusters & ~results.keep_speaker_sys_clusters & results.heuristic_speaker_resolution, conll_cols + nec_cols + mention_cols].mean()
    print("Model - Keep speakers")
    pr = []
    for metric in conll_cols + nec_cols + mention_cols:
        x = f"{100*(model_minus_keep_speakers[metric] - model[metric]):.1f}"
        print(f"\t{metric:20s} = {model_minus_keep_speakers[metric] - model[metric]:.3f}")
        pr.append(x)
    print(" & ".join(pr))
    print()

    model_minus_merge_clusters = results.loc[results.use_speaker_sep & ~results.keep_person_sys_clusters & results.keep_speaker_sys_clusters & ~results.heuristic_speaker_resolution, conll_cols + nec_cols + mention_cols].mean()
    print("Model - Merge clusters")
    pr = []
    for metric in conll_cols + nec_cols + mention_cols:
        x = f"{100*(model_minus_merge_clusters[metric] - model[metric]):.1f}"
        print(f"\t{metric:20s} = {model_minus_merge_clusters[metric] - model[metric]:.3f}")
        pr.append(x)
    print(" & ".join(pr))
    print()

    print("All combinations")
    config_df = results.groupby(config_cols)[conll_cols + nec_cols + mention_cols].mean()
    config_df.sort_values(by="conll2012_F1", inplace=True, ascending=False)
    print(config_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="print evaluation numbers", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--results", type=str, help="evaluation results file", default="../../results/acl21/coreference_evaluation.all.csv")
    print_evaluation_numbers(parser.parse_args().results)