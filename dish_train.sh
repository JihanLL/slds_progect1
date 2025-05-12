#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_addr="localhost" \
    --master_port='12345' \
    train.py \
    --seed 42 \
    --epochs 35 \
    --learning_rate 0.001 \
    --batch_size 256 \
    --num_workers 16 \
    --model "CNN" \
    --L1_parameter 0.00001 \
    --L2_parameter 0.00001 \
    --momentum 0.9 \
    --full_dataset \
    --flopanalysis