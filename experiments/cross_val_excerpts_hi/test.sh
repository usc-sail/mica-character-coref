#!/bin/bash

python coref/movie_coref/run.py \
    --input_type=regular \
    --test_movie=avengers_endgame \
    --bert_lr=2e-5 \
    --character_lr=2e-4 \
    --coref_lr=2e-4 \
    --warmup_steps=50 \
    --train_excerpts \
    --hierarchical \
    --max_epochs=1 \
    --train_document_len=5120 \
    --train_overlap_len=0 \
    --dev_document_len=5120 \
    --dev_overlap_len=512 \
    --weight_decay=1e-3 \
    --dropout=0 \
    --test_document_lens=5120 \
    --test_overlap_lens=512 \
    --noeval_train \
    --n_epochs_no_eval=0 \
    --save_log \
    --save_predictions