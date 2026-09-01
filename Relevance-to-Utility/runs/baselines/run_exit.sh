#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
python summarizer/exit/run.py \
    --dataset_name 'hotpotqa' \
    --cache_name search_cache_500