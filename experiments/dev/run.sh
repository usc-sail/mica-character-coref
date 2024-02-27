#!/bin/bash

# pkill -9 python -u sbaruah

# input type
# python coref/movie_coref/run.py --save_log --save_predictions --save_loss_curve --input_type=regular
# python coref/movie_coref/run.py --save_log --save_predictions --save_loss_curve --input_type=nocharacters
# python coref/movie_coref/run.py --save_log --save_predictions --save_loss_curve --input_type=addsays

# filter and singleton
# python coref/movie_coref/run.py --save_log --save_predictions --save_loss_curve --input_type=addsays --filter_by_cr
# python coref/movie_coref/run.py --save_log --save_predictions --save_loss_curve --input_type=addsays --noremove_singleton_cr --n_epochs_no_eval=2
# python coref/movie_coref/run.py --save_log --save_predictions --save_loss_curve --input_type=addsays --filter_by_cr --noremove_singleton_cr

# # tune learning rate, warmup, weight decay, dropout
# bert_lrs=(1e-5 2e-5 5e-5)
# coref_cr_lrs=(1e-4 2e-4)
# warmups=(-1 0 1)
# weight_decays=(0 1e-3)
# dropouts=(0 0.3)

# for bert_lr in ${bert_lrs[@]}; do
# for coref_cr_lr in ${coref_cr_lrs[@]}; do
# for warmup in ${warmups[@]}; do
# for weight_decay in ${weight_decays[@]}; do
# for dropout in ${dropouts[@]}; do
#     python coref/movie_coref/run.py --save_log --save_predictions --save_loss_curve --input_type=regular \
#         --bert_lr=$bert_lr --coref_lr=$coref_cr_lr --character_lr=$coref_cr_lr --warmup=$warmup \
#         --weight_decay=$weight_decay --dropout=$dropout
# done; done; done; done; done;

# missed configs
# python coref/movie_coref/run.py --save_log --save_predictions --save_loss_curve --input_type=regular --bert_lr=5e-5 --coref_lr=2e-4 --character_lr=2e-4 --warmup=1 --weight_decay=0 --dropout=0
# python coref/movie_coref/run.py --save_log --save_predictions --save_loss_curve --input_type=regular --bert_lr=5e-5 --coref_lr=2e-4 --character_lr=2e-4 --warmup=1 --weight_decay=1e-3 --dropout=0

# # further fine tuning
# bert_lrs=(2e-5 3e-5 4e-5 5e-5)
# weight_decays=(0 1e-4 1e-3 1e-2 1e-1)
# for bert_lr in ${bert_lrs[@]}; do
# for weight_decay in ${weight_decays[@]}; do
#     python coref/movie_coref/run.py --save_log --save_predictions --save_loss_curve \
#         --input_type=$1 --bert_lr=$bert_lr --coref_lr=2e-4 --character_lr=2e-4 --warmup=-1 --weight_decay=$weight_decay --dropout=0
# done; done;

# # genre fine tuning
# genres=(bc bn mz nw pt tc wb)
# for genre in ${genres[@]}; do
#     python coref/movie_coref/run.py --save_log --save_predictions --save_loss_curve \
#         --input_type=$1 --bert_lr=2e-5 --coref_lr=2e-4 --character_lr=2e-4 --warmup=-1 --weight_decay=1e-3 --dropout=0 --genre=$genre
# done;

# # train document length finetuning
# lens=(1024 2048 3072 4096 5120)
# for len in ${lens[@]}; do
#     python coref/movie_coref/run.py --save_log --save_predictions --save_loss_curve \
#         --input_type=$1 --bert_lr=2e-5 --coref_lr=2e-4 --character_lr=2e-4 --warmup=-1 --weight_decay=1e-3 --dropout=0 --train_document_len=$len
# done;

# # add/ don't add cr_to_coarse
# python coref/movie_coref/run.py --save_log --save_predictions --save_loss_curve \
#     --input_type=$1 --bert_lr=2e-5 --coref_lr=2e-4 --character_lr=2e-4 --warmup=-1 --weight_decay=1e-3 --dropout=0 --noadd_cr_to_coarse

# character recognition model finetuning
# cr_seq_lens=(256 512)
# gru_hidden_sizes=(128 256 512 1024)
# for cr_seq_len in ${cr_seq_lens[@]}; do
# for gru_hidden_size in ${gru_hidden_sizes[@]}; do
#     python coref/movie_coref/run.py --save_log --save_predictions --save_loss_curve \
#         --input_type=$1 --bert_lr=2e-5 --coref_lr=2e-4 --character_lr=2e-4 --warmup=-1 --weight_decay=1e-3 --dropout=0 --cr_seq_len=$cr_seq_len --gru_hidden_size=$gru_hidden_size
#     python coref/movie_coref/run.py --save_log --save_predictions --save_loss_curve \
#         --input_type=$1 --bert_lr=2e-5 --coref_lr=2e-4 --character_lr=2e-4 --warmup=-1 --weight_decay=1e-3 --dropout=0 --cr_seq_len=$cr_seq_len --gru_hidden_size=$gru_hidden_size --nogru_bi
# done; done

# cross validation leave-one-movie-out
python coref/movie_coref/run.py \
    --test_movie=avengers_endgame \
    --test_document_lens=5120 \
    --test_overlap_lens=0 \
    --test_overlap_lens=256 \
    --n_epochs_no_eval=-1 \
    --noeval_train \
    --save_log \
    --save_predictions \