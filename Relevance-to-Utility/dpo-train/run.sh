#!/bin/bash


export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch --config_file ./config/accelerate_configs/deepspeed_zero3.yaml \
    --main_process_port 29600 \
    --num_processes 4 \
    train.py \
    ./config/dpo/config_dpo.yaml
