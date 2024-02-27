#!/bin/bash

python coref/movie_coref/run.py \
    --output_dir=/scratch1/sbaruah/mica_text_coref/movie_coref/results/coreference/litbank_Jan15/ \
    --bert_lr=2e-5 \
    --character_lr=2e-4 \
    --coref_lr=2e-4 \
    --warmup_steps=50 \
    --litbank \
    --litbank_fold=0 \
    --max_epochs=1 \
    --repk=3 \
    --hierarchical \
    --genre=bn \
    --dev_merge_strategy=avg \
    --weight_decay=1e-3 \
    --dropout=0 \
    --patience=1 \
    --test_document_lens=512 \
    --test_overlap_lens=128 \
    --test_overlap_lens=64 \
    --test_merge_strategies=avg \
    --noeval_train \
    --save_log \
    --save_predictions \
    --save_loss_curve