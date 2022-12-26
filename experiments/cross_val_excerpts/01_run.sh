#!/bin/bash

preprocess_arr=(regular addsays nocharacters)
bert_lr_arr=(1e-5 2e-5 5e-5)

for preprocess in ${preprocess_arr[@]}; do
for bert_lr in ${bert_lr_arr[@]}; do
    sbatch carc/job.sh experiments/cross_val_excerpts/01.sh $preprocess $bert_lr
done; done;