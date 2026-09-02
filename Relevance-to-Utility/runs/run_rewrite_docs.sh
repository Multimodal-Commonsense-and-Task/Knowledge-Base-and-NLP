#!/bin/bash

dataset_name="hotpotqa"
model_name="/data/models/Llama-3.3-70B-Instruct"
search_cache_name="search_cache_500.json"
top_k=10
max_tokens=32768
context_length=2
subset_num=10000

subset_start_idx=0
echo "Start $model_name"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python scripts/run_rewrite_docs.py \
    --dataset_name $dataset_name \
    --split test \
    --top_k $top_k \
    --max_tokens $max_tokens \
    --search_cache_name $search_cache_name \
    --model_path $model_name \
    --subset_num $subset_num \
    --subset_start_idx $subset_start_idx \
    --context_length $context_length \
    --max_num_seqs 64 \
    --mode "conditional_cot2"
