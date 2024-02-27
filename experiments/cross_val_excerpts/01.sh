#!/bin/bash
# preprocess_arr=(regular addsays nocharacters)
# bert_lr_arr=(1e-5 2e-5 5e-5)

preprocess=$1
bert_lr=$2

model_lr_arr=(1e-4 2e-4 5e-4)
warmup_arr=(-1 0 50 100)
test_movie_arr=(avengers_endgame dead_poets_society john_wick prestige quiet_place zootopia)

for test_movie in ${test_movie_arr[@]}; do
for model_lr in ${model_lr_arr[@]}; do
for warmup in ${warmup_arr[@]}; do
    python coref/movie_coref/run.py \
        --input_type=$preprocess \
        --test_movie=$test_movie \
        --bert_lr=$bert_lr \
        --character_lr=$model_lr \
        --coref_lr=$model_lr \
        --warmup_steps=$warmup \
        --train_excerpts \
        --max_epochs=20 \
        --train_document_len=5120 \
        --train_overlap_len=0 \
        --dev_document_len=5120 \
        --dev_overlap_len=512 \
        --dev_merge_strategy=avg \
        --weight_decay=1e-3 \
        --dropout=0 \
        --test_document_lens=5120 \
        --test_overlap_lens=512 \
        --test_merge_strategies=avg \
        --eval_train \
        --save_log \
        --save_predictions \
        --save_loss_curve
done; done; done;