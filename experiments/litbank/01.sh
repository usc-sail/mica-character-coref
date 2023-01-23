#!/bin/bash

fold=$1
bert_lr_arr=(1e-5 2e-5 5e-5)
model_lr_arr=(1e-4 2e-4 5e-4)

py="python coref/movie_coref/run.py \
        --output_dir=/scratch1/sbaruah/mica_text_coref/litbank/Jan16/ \
        --litbank \
        --litbank_fold=$fold \
        --warmup_steps=-1 \
        --max_epochs=20 \
        --genre=wb \
        --repk=3 \
        --weight_decay=0 \
        --dropout=0 \
        --patience=5 \
        --test_document_lens=512 \
        --test_document_lens=1024 \
        --test_overlap_lens=128 \
        --test_overlap_lens=64 \
        --test_merge_strategies=avg \
        --test_merge_strategies=pre \
        --test_merge_strategies=post \
        --eval_train \
        --save_log \
        --save_predictions \
        --save_loss_curve"

for bert_lr in ${bert_lr_arr[@]}; do
    for model_lr in ${model_lr_arr[@]}; do
        $py --bert_lr=$bert_lr --coref_lr=$model_lr --character_lr=$model_lr
    done;
done;