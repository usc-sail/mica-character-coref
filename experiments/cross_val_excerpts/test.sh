#!/bin/bash
# Test lomo excerpts training

python coref/movie_coref/run.py \
    --test_movie=avengers_endgame \
    --train_excerpts \
    --max_epochs=1 \
    --test_document_lens=4096 \
    --test_document_lens=5120 \
    --test_overlap_lens=256 \
    --test_overlap_lens=512 \
    --test_merge_strategies=avg \
    --test_merge_strategies=max \
    --test_merge_strategies=min \
    --save_log