#!/bin/bash

weight_decay_arr=(0 1e-4 1e-3 1e-2 1e-1 1 10 100)

for weight_decay in ${weight_decay_arr[@]}; do
    sbatch carc/job.sh experiments/cross_val_excerpts/02.sh $weight_decay
done;