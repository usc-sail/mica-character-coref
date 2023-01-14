#!/bin/bash

document_lens=(2048 3072 4096 5120 8192 10240)

for document_len in ${document_lens[@]}; do
    sbatch carc/job.sh experiments/final/01.sh $document_len
    sbatch carc/job.sh experiments/final/02.sh $document_len
done;