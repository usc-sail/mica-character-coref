#!/bin/bash
# Missed experiments

elems=(
"nocharacters 5e-05 0.0001 0 zootopia"
"addsays 2e-05 0.0001 -1 avengers_endgame"
"nocharacters 1e-05 0.0001 -1 dead_poets_society"
"regular 5e-05 0.0001 -1 avengers_endgame"
"regular 2e-05 0.0001 -1 avengers_endgame"
"addsays 1e-05 0.0001 -1 dead_poets_society"
"regular 1e-05 0.0001 -1 dead_poets_society"
"nocharacters 5e-05 0.0001 -1 avengers_endgame"
"addsays 5e-05 0.0001 -1 avengers_endgame"
"regular 2e-05 0.0001 -1 dead_poets_society"
"nocharacters 2e-05 0.0001 -1 avengers_endgame"
"nocharacters 1e-05 0.0001 -1 avengers_endgame"
"nocharacters 5e-05 0.0001 -1 dead_poets_society"
"addsays 1e-05 0.0001 -1 avengers_endgame"
)

for elem in "${elems[@]}"; do
    read -a strarr <<< $elem
    echo ${strarr[0]} ${strarr[1]} ${strarr[2]} ${strarr[3]} ${strarr[4]}
    python coref/movie_coref/run.py \
        --output_dir=/scratch1/sbaruah/mica_text_coref/movie_coref/results/coreference/cross_val_excerpts_Dec19-21 \
        --input_type=${strarr[0]} \
        --test_movie=${strarr[4]} \
        --bert_lr=${strarr[1]} \
        --character_lr=${strarr[2]} \
        --coref_lr=${strarr[2]} \
        --warmup_steps=${strarr[3]} \
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
done