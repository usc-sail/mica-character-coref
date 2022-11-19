#!/bin/bash

preprocess_vals=("none" "addsays" "nocharacters")
genres=("bc" "bn" "mz" "nw" "pt" "tc" "wb")
split_lens=("2048" "3072" "4096" "5120")
overlap_lens=("128" "256" "512")
entity_vals=("all" "person" "speaker")

merge=("" "no")
gold_mention=("" "no")
gold_singleton=("" "no")
strategy_arr=("none" "max" "min" "before" "after" "average")
device=2

for preprocess in "${preprocess_vals[@]}"; do
for genre in "${genres[@]}"; do
for split_len in "${split_lens[@]}"; do
for overlap_len in "${overlap_lens[@]}"; do
for entity in "${entity_vals[@]}"; do
for mg in "${merge[@]}"; do
for gm in "${gold_mention[@]}"; do
for gs in "${gold_singleton[@]}"; do
    pids=()
    i=0
    for strategy in "${strategy_arr[@]}"; do
        if [ -z "$mg" ]; then mgx="yes"; else mgx="no"; fi
        if [ -z "$gm" ]; then gmx="yes"; else gmx="no"; fi
        if [ -z "$gs" ]; then gsx="yes"; else gsx="no"; fi
        echo -e "preprocess=${preprocess} genre=${genre} split_len=${split_len} overlap_len=${overlap_len} entity=${entity}"
        echo -e "merge_speakers=${mgx} provide_gold_mentions=${gmx} remove_gold_singletons=${gsx} merge_strategy=${strategy}"
        CUDA_VISIBLE_DEVICES=$device python coref/movie_coref/baseline_main.py --preprocess=$preprocess --wl_genre=$genre --split_len=$split_len --overlap_len=$overlap_len \
            --${mg}merge_speakers --${gm}provide_gold_mentions --${gs}remove_gold_singletons --merge_strategy=${strategy} \
            --nouse_reference_scorer --calc_results --run_train --use_gpu &
        pids[${i}]=$!
        i=$((i+1))
        device=$((5-device))
    done;
    echo "PIDS = ${pids[@]}"
    for pid in ${pids[*]}; do
        echo "waiting for ${pid}"
        wait $pid
    done
done; done; done; done; done; done; done; done;