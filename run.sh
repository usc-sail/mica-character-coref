#!/bin/bash

sudo pkill -9 python
NUMEXPR_MAX_THREADS=8 accelerate launch --num_cpu_threads_per_process=8 \
    coref/seq_coref/train_main.py \
    --train_batch_size=8 \
    --infer_batch_size=1 \
    --grad_accumulation_steps=1 \
    --use_grad_checkpointing \
    --use_mixed_precision