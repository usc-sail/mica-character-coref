#!/bin/bash

document_len=$1
test_movie_arr=(avengers_endgame dead_poets_society john_wick prestige quiet_place zootopia)
repks=(1 2 3 4 5)

py="python coref/movie_coref/run.py \
        --output_dir=/scratch1/sbaruah/mica_text_coref/movie_coref/results/coreference/final_Jan12/ \
        --input_type=regular \
        --bert_lr=2e-5 \
        --character_lr=2e-4 \
        --coref_lr=2e-4 \
        --warmup_steps=50 \
        --train_excerpts \
        --max_epochs=20 \
        --genre=bn \
        --train_document_len=5120 \
        --train_overlap_len=0 \
        --dev_merge_strategy=avg \
        --weight_decay=1e-3 \
        --dropout=0 \
        --test_merge_strategies=avg \
        --noeval_train \
        --save_log \
        --save_predictions \
        --save_loss_curve"

for test_movie in ${test_movie_arr[@]}; do
    for repk in ${repks[@]}; do
        $py --test_movie=$test_movie --hierarchical \
            --dev_document_len=$document_len --test_document_lens=$document_len --repk=$repk
    done;
done;