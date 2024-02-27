#!/bin/bash

dev_document_len_arr=(2048 3072 4096 5120 8192 10240 20480)

for dev_document_len in ${dev_document_len_arr[@]}; do
    sbatch carc/job.sh experiments/cross_val_excerpts/03.sh $dev_document_len
done;