#!/bin/bash
# dev_document_len_arr=(5120 8192 10240 20480)

dev_document_len=$1
dev_overlap_len_arr=(2048 3072 4096 5120)
test_movie_arr=(avengers_endgame dead_poets_society john_wick prestige quiet_place zootopia)

for test_movie in ${test_movie_arr[@]}; do
for dev_overlap_len in ${dev_overlap_len_arr[@]}; do
    python coref/movie_coref/run.py \
        --output_dir=/scratch1/sbaruah/mica_text_coref/movie_coref/results/coreference/cross_val_excerpts_Dec28/ \
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
        --save_loss_curve
done; done;