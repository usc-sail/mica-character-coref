import argparse
import pandas as pd

def print_evaluation_numbers(results_file):
    results = pd.read_csv(results_file, index_col=None)
    results = results[~results.keep_only_speaker_gold_clusters & ~results.remove_singleton_gold_clusters & ~results.heuristic_pronoun_resolution]
    metric_cols = ["mention_P", "mention_R", "mention_F1", "muc_F1", "bcubed_F1", "ceafe_F1", "conll2012_F1", "nec_per_chains_missed", "nec_F1", "nec_name_F1", "nec_pronoun_F1", "nec_nominal_F1"]
    hparam_cols = ["min_speaker_sim", "max_speaker_merges"]

    for hparam_col in hparam_cols:
        values = results[hparam_col].unique()
        if len(values) == 1:
            print(f"{hparam_col} = {values[0]}")
        else:
            print(f"{hparam_col} = {values}")
    print()

    print("Baseline")
    baseline = results.loc[~results.use_speaker_sep & results.keep_person_sys_clusters & ~results.keep_speaker_sys_clusters & ~results.heuristic_speaker_resolution, metric_cols].mean()
    for metric in metric_cols:
        print(f"\t{metric:25s} = {baseline[metric]:.3f}")
    print()
    
    print("Model")
    model = results.loc[results.use_speaker_sep & ~results.keep_person_sys_clusters & results.keep_speaker_sys_clusters & results.heuristic_speaker_resolution, metric_cols].mean()
    for metric in metric_cols:
        print(f"\t{metric:25s} = {model[metric]:.3f}")
    print()

    print("Model - Add says")
    model1 = results.loc[~results.use_speaker_sep & ~results.keep_person_sys_clusters & results.keep_speaker_sys_clusters & results.heuristic_speaker_resolution, metric_cols].mean()
    delta1 = model1 - model
    for metric in metric_cols:
        print(f"\t{metric:25s} = {model1[metric]:.3f} ({delta1[metric]:.4f})")
    print()

    print("Model - Keep speakers")
    model2 = results.loc[results.use_speaker_sep & results.keep_person_sys_clusters & ~results.keep_speaker_sys_clusters & results.heuristic_speaker_resolution, metric_cols].mean()
    delta2 = model2 - model
    for metric in metric_cols:
        print(f"\t{metric:25s} = {model2[metric]:.3f} ({delta2[metric]:.4f})")
    print()

    print("Model - Merge clusters")
    model3 = results.loc[results.use_speaker_sep & ~results.keep_person_sys_clusters & results.keep_speaker_sys_clusters & ~results.heuristic_speaker_resolution, metric_cols].mean()
    delta3 = model3 - model
    for metric in metric_cols:
        print(f"\t{metric:25s} = {model3[metric]:.3f} ({delta3[metric]:.4f})")
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="print evaluation numbers", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--results", type=str, help="evaluation results file", default="results/coreference_evaluation.csv")
    print_evaluation_numbers(parser.parse_args().results)