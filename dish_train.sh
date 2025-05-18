#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_addr="localhost" \
    --master_port='12945' \
    train.py \
    --seed 42 \
    --epochs 80 \
    --dataset DDR \
    --ddr_root_dir data/DDR \
    --optimizer adam \
    --learning_rate 0.001 \
    --batch_size 128 \
    --num_workers 12 \
    --model CNNv4 \
    --L1_parameter 0.00001 \
    --L2_parameter 0.00001 \
    --momentum 0.9 \
    --full_dataset \