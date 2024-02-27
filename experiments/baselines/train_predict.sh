#!/bin/bash

preprocess_vals=("none" "addsays" "nocharacters")
genres=("bc" "bn" "mz" "nw" "pt" "tc" "wb")
split_lens=("5120" "4096" "3072" "2048")
overlap_lens=("512" "256" "128")

for preprocess in "${preprocess_vals[@]}"; do
for genre in "${genres[@]}"; do
for split_len in "${split_lens[@]}"; do
for overlap_len in "${overlap_lens[@]}"; do
    echo -e "preprocess=${preprocess} genre=${genre} split_len=${split_len} overlap_len=${overlap_len}"
    python coref/movie_coref/baseline_main.py --preprocess=$preprocess --wl_genre=$genre --split_len=$split_len --overlap_len=$overlap_len --overwrite --nocalc_results --run_train --use_gpu
done; done; done; done;