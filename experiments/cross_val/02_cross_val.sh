#!/bin/bash
# Run cross validation experiments that could not be completed by 01_cross_val.sh

tups=(
    "addsays 5e-5 1e-4 0.25 10240 john_wick"
    "addsays 5e-5 1e-4 1.00 10240 dead_poets_society"
    "addsays 5e-5 1e-4 1.00 20480 john_wick"
    "addsays 5e-5 2e-4 0.00 10240 avengers_endgame"
    "addsays 5e-5 2e-4 0.00 20480 avengers_endgame"
    "regular 5e-5 2e-4 0.50 10240 zootopia"
    "regular 5e-5 2e-4 0.50 20480 zootopia"
)

for tup in "${tups[@]}"; do
    read -a args <<< "$tup"
    preprocess=${args[0]}
    bert_lr=${args[1]}
    model_lr=${args[2]}
    warmup=${args[3]}
    document_len=${args[4]}
    test_movie=${args[5]}
    python coref/movie_coref/run.py \
        --test_movie=$test_movie \
        --test_document_lens=$document_len \
        --test_overlap_lens=512 \
        --test_overlap_lens=1024 \
        --test_overlap_lens=2048 \
        --test_merge_strategies=avg \
        --input_type=$preprocess \
        --bert_lr=$bert_lr \
        --character_lr=$model_lr \
        --coref_lr=$model_lr \
        --warmup=$warmup \
        --weight_decay=1e-3 \
        --dropout=0 \
        --max_epochs=20 \
        --patience=5 \
        --train_document_len=5120 \
        --add_cr_to_coarse \
        --n_epochs_no_eval=0 \
        --noeval_train \
        --save_log \
        --save_predictions \
        --save_loss_curve
done