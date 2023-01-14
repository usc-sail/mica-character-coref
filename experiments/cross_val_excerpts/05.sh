#!/bin/bash
# load_bert, freeze_bert, merge_strategy, add_cr_to_coarse, filter_by_cr, remove_singleton_cr

dev_document_len=$1
dev_overlap_len=$2
test_movie_arr=(avengers_endgame dead_poets_society john_wick prestige quiet_place zootopia)
merge_strategy_arr=(none pre post max min avg)

py="python coref/movie_coref/run.py \
        --output_dir=/scratch1/sbaruah/mica_text_coref/movie_coref/results/coreference/cross_val_excerpts_Dec30/ \
        --input_type=regular \
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
        --weight_decay=1e-3 \
        --dropout=0 \
        --test_document_lens=$dev_document_len \
        --test_overlap_lens=$dev_overlap_len \
        --noeval_train \
        --save_log \
        --save_predictions \
        --save_loss_curve"
py2="$py --dev_merge_strategy=avg --test_merge_strategies=avg"

for test_movie in ${test_movie_arr[@]}; do
    $py2 --noload_bert --test_movie=$test_movie
    $py2 --freeze_bert --test_movie=$test_movie
    $py2 --noadd_cr_to_coarse --test_movie=$test_movie
    $py2 --filter_by_cr --test_movie=$test_movie
    $py2 --noremove_singleton_cr --test_movie=$test_movie
    for merge_strategy in ${merge_strategy_arr[@]}; do
        $py --dev_merge_strategy=$merge_strategy --test_merge_strategies=$merge_strategy --test_movie=$test_movie
    done;
done;