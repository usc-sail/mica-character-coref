#!/bin/bash

genre=$1
preprocess_vals=("addsays" "none" "nocharacters")
split_lens=("2048" "3072" "4096" "5120")
overlap_lens=("128" "256" "512")

for preprocess in "${preprocess_vals[@]}"; do
for split_len in "${split_lens[@]}"; do
for overlap_len in "${overlap_lens[@]}"; do
    python coref/movie_coref/baseline_main.py \
        --preprocess=$preprocess \
        --genre=$genre \
        --split_len=$split_len \
        --overlap_len=$overlap_len \
        --calc_results \
        --lea
done; done; done