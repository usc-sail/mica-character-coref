#!/bin/bash

dev_document_len_arr=(5120 8192 10240 20480)

for dev_document_len in ${dev_document_len_arr[@]}; do
    sbatch carc/job.sh experiments/cross_val_excerpts/04.sh $dev_document_len
done;