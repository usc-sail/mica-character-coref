#!/bin/bash

data_dir=$1
test_movies=(avengers_endgame dead_poets_society john_wick prestige quiet_place zootopia)

# for movie in ${test_movies[@]}; do
#     python run.py --data_dir=$data_dir --test_movie=$movie --freeze_bert
#     python run.py --data_dir=$data_dir --test_movie=$movie --hierarchical --freeze_bert
# done

python train.py --data_dir=$data_dir --save --max_epochs=8