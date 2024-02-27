#!/bin/bash
# load_bert, freeze_bert, merge_strategy, add_cr_to_coarse, filter_by_cr, remove_singleton_cr
# missing experiments

py="python coref/movie_coref/run.py \
        --output_dir=/scratch1/sbaruah/mica_text_coref/movie_coref/results/coreference/cross_val_excerpts_Jan01/ \
        --input_type=regular \
        --bert_lr=2e-5 \
        --character_lr=2e-4 \
        --coref_lr=2e-4 \
        --warmup_steps=50 \
        --train_excerpts \
        --max_epochs=20 \
        --train_document_len=5120 \
        --train_overlap_len=0 \
        --weight_decay=1e-3 \
        --dropout=0 \
        --noeval_train \
        --save_log \
        --save_predictions \
        --save_loss_curve"

$py --dev_document_len=5120 --dev_overlap_len=512 --dev_merge_strategy=avg \
    --test_document_lens=5120 --test_overlap_lens=512 --test_merge_strategies=avg \
    --load_bert --nofreeze_bert --add_cr_to_coarse --nofilter_by_cr --remove_singleton_cr --test_movie=zootopia
$py --dev_document_len=5120 --dev_overlap_len=1024 --dev_merge_strategy=avg \
    --test_document_lens=5120 --test_overlap_lens=1024 --test_merge_strategies=avg \
    --load_bert --nofreeze_bert --add_cr_to_coarse --nofilter_by_cr --remove_singleton_cr --test_movie=zootopia
$py --dev_document_len=5120 --dev_overlap_len=1024 --dev_merge_strategy=max \
    --test_document_lens=5120 --test_overlap_lens=1024 --test_merge_strategies=max \
    --load_bert --nofreeze_bert --add_cr_to_coarse --nofilter_by_cr --remove_singleton_cr --test_movie=zootopia
$py --dev_document_len=5120 --dev_overlap_len=1024 --dev_merge_strategy=min \
    --test_document_lens=5120 --test_overlap_lens=1024 --test_merge_strategies=min \
    --load_bert --nofreeze_bert --add_cr_to_coarse --nofilter_by_cr --remove_singleton_cr --test_movie=zootopia
$py --dev_document_len=5120 --dev_overlap_len=2048 --dev_merge_strategy=avg \
    --test_document_lens=5120 --test_overlap_lens=2048 --test_merge_strategies=avg \
    --noload_bert --nofreeze_bert --add_cr_to_coarse --nofilter_by_cr --remove_singleton_cr --test_movie=zootopia
$py --dev_document_len=5120 --dev_overlap_len=2048 --dev_merge_strategy=avg \
    --test_document_lens=5120 --test_overlap_lens=2048 --test_merge_strategies=avg \
    --load_bert --nofreeze_bert --noadd_cr_to_coarse --nofilter_by_cr --remove_singleton_cr --test_movie=zootopia
$py --dev_document_len=5120 --dev_overlap_len=2048 --dev_merge_strategy=avg \
    --test_document_lens=5120 --test_overlap_lens=2048 --test_merge_strategies=avg \
    --load_bert --nofreeze_bert --add_cr_to_coarse --nofilter_by_cr --noremove_singleton_cr --test_movie=zootopia
$py --dev_document_len=5120 --dev_overlap_len=2048 --dev_merge_strategy=avg \
    --test_document_lens=5120 --test_overlap_lens=2048 --test_merge_strategies=avg \
    --load_bert --nofreeze_bert --add_cr_to_coarse --nofilter_by_cr --remove_singleton_cr --test_movie=quiet_place
$py --dev_document_len=5120 --dev_overlap_len=2048 --dev_merge_strategy=avg \
    --test_document_lens=5120 --test_overlap_lens=2048 --test_merge_strategies=avg \
    --load_bert --nofreeze_bert --add_cr_to_coarse --nofilter_by_cr --remove_singleton_cr --test_movie=zootopia
$py --dev_document_len=5120 --dev_overlap_len=2048 --dev_merge_strategy=max \
    --test_document_lens=5120 --test_overlap_lens=2048 --test_merge_strategies=max \
    --load_bert --nofreeze_bert --add_cr_to_coarse --nofilter_by_cr --remove_singleton_cr --test_movie=zootopia
$py --dev_document_len=5120 --dev_overlap_len=2048 --dev_merge_strategy=min \
    --test_document_lens=5120 --test_overlap_lens=2048 --test_merge_strategies=min \
    --load_bert --nofreeze_bert --add_cr_to_coarse --nofilter_by_cr --remove_singleton_cr --test_movie=zootopia
$py --dev_document_len=5120 --dev_overlap_len=2048 --dev_merge_strategy=none \
    --test_document_lens=5120 --test_overlap_lens=2048 --test_merge_strategies=none \
    --load_bert --nofreeze_bert --add_cr_to_coarse --nofilter_by_cr --remove_singleton_cr --test_movie=zootopia
$py --dev_document_len=5120 --dev_overlap_len=2048 --dev_merge_strategy=post \
    --test_document_lens=5120 --test_overlap_lens=2048 --test_merge_strategies=post \
    --load_bert --nofreeze_bert --add_cr_to_coarse --nofilter_by_cr --remove_singleton_cr --test_movie=zootopia
$py --dev_document_len=5120 --dev_overlap_len=2048 --dev_merge_strategy=pre \
    --test_document_lens=5120 --test_overlap_lens=2048 --test_merge_strategies=pre \
    --load_bert --nofreeze_bert --add_cr_to_coarse --nofilter_by_cr --remove_singleton_cr --test_movie=zootopia
$py --dev_document_len=5120 --dev_overlap_len=2048 --dev_merge_strategy=avg \
    --test_document_lens=5120 --test_overlap_lens=2048 --test_merge_strategies=avg \
    --load_bert --nofreeze_bert --add_cr_to_coarse --filter_by_cr --remove_singleton_cr --test_movie=zootopia
$py --dev_document_len=5120 --dev_overlap_len=2048 --dev_merge_strategy=avg \
    --test_document_lens=5120 --test_overlap_lens=2048 --test_merge_strategies=avg \
    --load_bert --freeze_bert --add_cr_to_coarse --nofilter_by_cr --remove_singleton_cr --test_movie=zootopia
$py --dev_document_len=8192 --dev_overlap_len=2048 --dev_merge_strategy=avg \
    --test_document_lens=8192 --test_overlap_lens=2048 --test_merge_strategies=avg \
    --load_bert --nofreeze_bert --add_cr_to_coarse --nofilter_by_cr --remove_singleton_cr --test_movie=zootopia
$py --dev_document_len=8192 --dev_overlap_len=3072 --dev_merge_strategy=avg \
    --test_document_lens=8192 --test_overlap_lens=3072 --test_merge_strategies=avg \
    --load_bert --nofreeze_bert --add_cr_to_coarse --nofilter_by_cr --noremove_singleton_cr --test_movie=zootopia
$py --dev_document_len=8192 --dev_overlap_len=3072 --dev_merge_strategy=avg \
    --test_document_lens=8192 --test_overlap_lens=3072 --test_merge_strategies=avg \
    --load_bert --nofreeze_bert --add_cr_to_coarse --nofilter_by_cr --remove_singleton_cr --test_movie=zootopia
$py --dev_document_len=8192 --dev_overlap_len=3072 --dev_merge_strategy=max \
    --test_document_lens=8192 --test_overlap_lens=3072 --test_merge_strategies=max \
    --load_bert --nofreeze_bert --add_cr_to_coarse --nofilter_by_cr --remove_singleton_cr --test_movie=zootopia
$py --dev_document_len=8192 --dev_overlap_len=3072 --dev_merge_strategy=min \
    --test_document_lens=8192 --test_overlap_lens=3072 --test_merge_strategies=min \
    --load_bert --nofreeze_bert --add_cr_to_coarse --nofilter_by_cr --remove_singleton_cr --test_movie=zootopia
$py --dev_document_len=8192 --dev_overlap_len=3072 --dev_merge_strategy=none \
    --test_document_lens=8192 --test_overlap_lens=3072 --test_merge_strategies=none \
    --load_bert --nofreeze_bert --add_cr_to_coarse --nofilter_by_cr --remove_singleton_cr --test_movie=zootopia
$py --dev_document_len=8192 --dev_overlap_len=3072 --dev_merge_strategy=post \
    --test_document_lens=8192 --test_overlap_lens=3072 --test_merge_strategies=post \
    --load_bert --nofreeze_bert --add_cr_to_coarse --nofilter_by_cr --remove_singleton_cr --test_movie=zootopia
$py --dev_document_len=8192 --dev_overlap_len=3072 --dev_merge_strategy=pre \
    --test_document_lens=8192 --test_overlap_lens=3072 --test_merge_strategies=pre \
    --load_bert --nofreeze_bert --add_cr_to_coarse --nofilter_by_cr --remove_singleton_cr --test_movie=zootopia