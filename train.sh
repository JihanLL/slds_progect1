#!/bin/bash
torchrun --nproc_per_node=4 main.py \
    --seed 42 \
    --epochs 35 \
    --learning_rate 0.001 \
    --batch_size 128 \
    --num_workers 4 \
    --model "CNN" \
    --L1_parameter 0.00001 \
    --L2_parameter 0.00001 \
    --momentum 0.9 \
    --full_dataset True