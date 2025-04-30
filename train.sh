#!/bin/bash
python train.py \
    --seed 42 \
    --epochs 100 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --num_workers 4 \
    --model "CNN" \
    --L1_parameter 0.00001 \
    --L2_parameter 0.00001 \
    --momentum 0.9