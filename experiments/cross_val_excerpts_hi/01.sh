#!/bin/bash

dev_document_len=$1
repk_arr=(1 2 3 4)
test_movie_arr=(avengers_endgame dead_poets_society john_wick prestige quiet_place zootopia)

for repk in ${repk_arr[@]}; do
for test_movie in ${test_movie_arr[@]}; do
    python coref/movie_coref/run.py \
        --output_dir=/scratch1/sbaruah/mica_text_coref/movie_coref/results/coreference/cross_val_excerpts_hi_Dec29/ \
        --input_type=regular \
        --test_movie=$test_movie \
        --bert_lr=2e-5 \
        --character_lr=2e-4 \
        --coref_lr=2e-4 \
        --warmup_steps=50 \
        --train_excerpts \
        --hierarchical \
        --max_epochs=20 \
        --train_document_len=5120 \
        --train_overlap_len=0 \
        --dev_document_len=$dev_document_len \
        --repk=$repk \
        --weight_decay=1e-3 \
        --dropout=0 \
        --test_document_lens=$dev_document_len \
        --noeval_train \
        --save_log \
        --save_predictions \
        --save_loss_curve
done; done;