#!/bin/bash

preprocess_vals=("none" "addsays" "nocharacters")
entity_vals=("all" "person" "speaker")
genres=("bc" "bn" "mz" "nw" "pt" "tc" "wb")

for preprocess in "${preprocess_vals[@]}"
do
    for entity in "${entity_vals[@]}"
    do
        for genre in "${genres[@]}"
        do
            python coref/movie_coref/baseline_main.py --preprocess=$preprocess --entity=$entity --wl_genre=$genre --merge_speakers
            python coref/movie_coref/baseline_main.py --preprocess=$preprocess --entity=$entity --wl_genre=$genre --nomerge_speakers
        done
    done
done