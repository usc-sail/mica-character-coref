#!/bin/bash

preprocess_vals=("none" "addsays" "nocharacters")
entity_vals=("all" "person" "speaker")
genres=("bc" "bn" "mz" "nw" "pt" "tc" "wb")
merge=("" "no")
gold_mention=("" "no")
gold_singleton=("" "no")

rm -f data/movie_coref/results/coreference/baselines/train.baseline.tsv
echo -e "preprocess\tgenre\tentity\tmerge_speakers\tprovide_gold_mentions\tremove_gold_singletons\
\toverlap_len\tsplit_len\tmetric\tmovie\tprecision\trecall\tf1" > \
data/movie_coref/results/coreference/baselines/train.baseline.tsv
for preprocess in "${preprocess_vals[@]}"; do
for entity in "${entity_vals[@]}"; do
for genre in "${genres[@]}"; do
for mg in "${merge[@]}"; do
for gm in "${gold_mention[@]}"; do
for gs in "${gold_singleton[@]}"; do
if [ -z "$mg" ]; then mgx="yes"; else mgx="no"; fi
if [ -z "$gm" ]; then gmx="yes"; else gmx="no"; fi
if [ -z "$gs" ]; then gsx="yes"; else gsx="no"; fi
echo -e "preprocess=${preprocess} entity=${entity} genre=${genre} merge_speakers=${mgx} \
provide_gold_mentions=${gmx} remove_gold_singletons=${gsx}"
python coref/movie_coref/baseline_main.py --preprocess=$preprocess --entity=$entity \
--wl_genre=$genre --${mg}merge_speakers --${gm}provide_gold_mentions --${gs}remove_gold_singletons \
--append_results --run_train --split_len=5120 --overlap_len=512 --use_gpu --overwrite
exit
done; done; done; done; done; done