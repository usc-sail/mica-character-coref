#!/bin/bash

for fold in {0..9}; do
    sbatch carc/job.sh experiments/litbank/01.sh $fold
done;