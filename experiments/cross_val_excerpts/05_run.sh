#!/bin/bash

sbatch carc/job.sh experiments/cross_val_excerpts/05.sh 5120 512
sbatch carc/job.sh experiments/cross_val_excerpts/05.sh 5120 1024
sbatch carc/job.sh experiments/cross_val_excerpts/05.sh 5120 2048
sbatch carc/job.sh experiments/cross_val_excerpts/05.sh 8192 2048
sbatch carc/job.sh experiments/cross_val_excerpts/05.sh 8192 3072