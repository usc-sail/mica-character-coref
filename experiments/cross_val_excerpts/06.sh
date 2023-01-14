#!/bin/bash
# genre, bce_weight

dev_document_len=$1
dev_overlap_len=$2
test_movie_arr=(avengers_endgame dead_poets_society john_wick prestige quiet_place zootopia)
genre_arr=(bc bn mz nw pt tc wb)
bce_weight_arr=(0 0.25 0.5 0.75 1)

py="python coref/movie_coref/run.py \
        --output_dir=/scratch1/sbaruah/mica_text_coref/movie_coref/results/coreference/cross_val_excerpts_Jan02/ \
        --input_type=regular \
        --bert_lr=2e-5 \
        --character_lr=2e-4 \
        --coref_lr=2e-4 \
        --warmup_steps=50 \
        --train_excerpts \
        --max_epochs=20 \
        --train_document_len=5120 \
        --train_overlap_len=0 \
        --dev_document_len=$dev_document_len \
        --dev_overlap_len=$dev_overlap_len \
        --dev_merge_strategy=avg \
        --weight_decay=1e-3 \
        --dropout=0 \
        --test_document_lens=$dev_document_len \
        --test_overlap_lens=$dev_overlap_len \
        --test_merge_strategies=avg \
        --noeval_train \
        --save_log \
        --save_predictions \
        --save_loss_curve"

for test_movie in ${test_movie_arr[@]}; do
    for genre in ${genre_arr[@]}; do
        $py --test_movie=$test_movie --genre=$genre
    done;
    for bce_weight in ${bce_weight_arr[@]}; do
        $py --test_movie=$test_movie --bce_weight=$bce_weight
    done;
done;