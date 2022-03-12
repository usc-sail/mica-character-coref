import argparse
from random import random
import time
from evaluate_mention_identification import evaluate_mention_identification
import pandas as pd
from sklearn.model_selection import ParameterGrid
from allennlp.predictors.predictor import Predictor
from evaluate_by_joining_elements import evaluate_coreference_by_joining_elements
from evaluate_coreference_nec import evaluate_coreference_nec
import random

def evaluate_all_by_joining_elements(results_file, min_speaker_sim, max_speaker_merge, n_times):
    print("loading allennlp coreference model")
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz")

    names = ["basterds", "bourne", "shawshank"]
    unchanged_vars = ["keep_only_speaker_gold_clusters", "remove_singleton_gold_clusters", "heuristic_pronoun_resolution"]
    boolean_vars = ["use_speaker_sep", "keep_person_sys_clusters", "keep_speaker_sys_clusters", "heuristic_speaker_resolution"]
    hparams = ["min_speaker_sim", "max_speaker_merges"]
    permutation_cols = ["random_seed", "start_line_index", "end_line_index"]
    conll_eval_metrics = ["muc_R", "muc_P", "muc_F1", "bcubed_R", "bcubed_P", "bcubed_F1", "ceafe_R", "ceafe_P", "ceafe_F1", "conll2012_R", "conll2012_P", "conll2012_F1"]
    nec_eval_metrics = ["nec_F1", "nec_per_chains_missed", "nec_name_F1", "nec_pronoun_F1", "nec_nominal_F1"]
    mention_eval_metrics = ["mention_P", "mention_R", "mention_F1"]
    header = ["script"] + unchanged_vars + boolean_vars + hparams + permutation_cols + conll_eval_metrics + nec_eval_metrics + mention_eval_metrics

    records = []
    cache_document_to_coref_result = {}

    for i in range(n_times):
        random_seed = random.randint(0, 1000000)

        for j, name in enumerate(names):
            print(f"script = {name}\n")
            parsed_file = f"data/annotation/{name}.script_parsed.txt"
            coref_file = f"data/annotation/{name}.coref.mapped.csv"

            configurations_dict = dict([(var, [False]) for var in unchanged_vars] + [(var, [False, True]) for var in boolean_vars] + [("min_speaker_sim", [min_speaker_sim]), ("max_speaker_merges", [max_speaker_merge])] + [("random_seed", True)])
            configurations = list(ParameterGrid(configurations_dict))
            print(f"{len(configurations)} configurations")

            for k, configuration in enumerate(configurations):
                start = time.time()
                print(f"Experiment {i*len(names)*len(configurations) + j*len(configurations) + k}/{len(n_times)*len(names)*len(configurations)}")
                
                result = evaluate_coreference_by_joining_elements(parsed_file, coref_file, random_seed=random_seed, coreference_model=predictor, cache_document_to_coref_result=cache_document_to_coref_result, **configuration)
                conll_result = result["evaluation"]
                cache_document_to_coref_result[result["document"]] = result["coref_result"]
                nec_result = evaluate_coreference_nec(result["gold_clusters"], result["sys_clusters"], result["coref_dataframe"], result["document"])
                mention_result = evaluate_mention_identification(result["gold_clusters"].values(), result["sys_clusters"])

                record = [name] + [configuration[var] for var in unchanged_vars + boolean_vars + hparams]
                record += [random_seed, result["start_line_index"], result["end_line_index"]]

                for metric in conll_eval_metrics:
                    key, subkey = metric.split("_")
                    record.append(conll_result[key][subkey])
                
                for metric in nec_eval_metrics:
                    record.append(nec_result[metric])

                for metric in mention_eval_metrics:
                    record.append(mention_result[metric])

                records.append(record)

                delta = time.time() - start
                minutes, seconds = delta//60, delta%60
                print(f"time taken = {minutes} min {seconds} sec")
                print()
    
    result_df = pd.DataFrame(records, columns=header)
    result_df.to_csv(results_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="permutation test of coreference rules", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--results", type=str, help="csv file where the results will be saved", default="results/coreference_evaluation.permutation.csv")
    parser.add_argument("--min_speaker_similarity", type=float, help="minimum normalized lcs similarity (between 0 and 1) to cluster speaker names", default=0.6)
    parser.add_argument("--max_speaker_merges", type=int, help="maximum number of merges to perform for speaker clustering", default=3)
    parser.add_argument("--n_times", type=int, help="number of samples", default=100)
    args = parser.parse_args()
    
    results_file = args.results
    min_speaker_sim = args.min_speaker_similarity
    max_speaker_merges = args.max_speaker_merges
    n_times = args.n_times
    evaluate_all_by_joining_elements(results_file, min_speaker_sim, max_speaker_merges)