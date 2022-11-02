#!/bin/bash

preprocess_vals=("none" "addsays" "nocharacters")
entity_vals=("all" "person" "speaker")
genres=("bc" "bn" "mz" "nw" "pt" "tc" "wb")
merge=("" "no")
gold_mention=("" "no")
gold_singleton=("" "no")

rm -f data/movie_coref/results/coreference/baselines/baseline.tsv
echo -e "preprocess\tgenre\tentity\tmerge_speakers\tprovide_gold_mentions\tremove_gold_singletons\
\tmuc_precision\tmuc_recall\tmuc_f1\tb_cubed_precision\tb_cubed_recall\tb_cubed_f1\
\tceafe_precision\tceafe_recall\tceafe_f1\taverage_f1" > \
data/movie_coref/results/coreference/baselines/baseline.tsv
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
--wl_genre=$genre --${mg}merge_speakers --${gm}provide_gold_mentions --${gs}remove_gold_singletons
done; done; done; done; done; done