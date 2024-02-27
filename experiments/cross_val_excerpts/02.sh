#!/bin/bash
# weight_decay_arr=(0 1e-4 1e-3 1e-2 1e-1 1 10 100)

weight_decay=$1
dropout_arr=(0 0.2 0.4 0.6 0.8)
test_movie_arr=(avengers_endgame dead_poets_society john_wick prestige quiet_place zootopia)

for dropout in ${dropout_arr[@]}; do
for test_movie in ${test_movie_arr[@]}; do
    python coref/movie_coref/run.py \
        --input_type=regular \
        --test_movie=$test_movie \
        --bert_lr=2e-5 \
        --character_lr=2e-4 \
        --coref_lr=2e-4 \
        --warmup_steps=50 \
        --train_excerpts \
        --max_epochs=20 \
        --train_document_len=5120 \
        --train_overlap_len=0 \
        --dev_document_len=5120 \
        --dev_overlap_len=512 \
        --dev_merge_strategy=avg \
        --weight_decay=$weight_decay \
        --dropout=$dropout \
        --test_document_lens=5120 \
        --test_overlap_lens=512 \
        --test_merge_strategies=avg \
        --noeval_train \
        --save_log \
        --save_predictions \
        --save_loss_curve
done; done;