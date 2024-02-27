#!/bin/bash
# Run cross validation experiments with different hyperparameters with fixed dev set (dev set = 3 short excerpts)

if [ $# -lt 2 ]; then
    echo -e "Usage:\n./cross_val.sh MOVIE DOCUMENT_LEN"
    exit
fi

movie=$1
document_len=$2
preprocess_arr=(regular addsays)
bert_lrs=(1e-5 2e-5 5e-5)
coref_cr_lrs=(1e-4 2e-4)
warmups=(-1 0 0.25 0.5 1)

for preprocess in ${preprocess_arr[@]}; do
for bert_lr in ${bert_lrs[@]}; do
for coref_cr_lr in ${coref_cr_lrs[@]}; do
for warmup in ${warmups[@]}; do
    python coref/movie_coref/run.py \
        --test_movie=$movie \
        --test_document_lens=$document_len \
        --test_overlap_lens=512 \
        --test_overlap_lens=1024 \
        --test_overlap_lens=2048 \
        --test_merge_strategies=avg \
        --input_type=$preprocess \
        --bert_lr=$bert_lr \
        --character_lr=$coref_cr_lr \
        --coref_lr=$coref_cr_lr \
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
done; done; done; done;