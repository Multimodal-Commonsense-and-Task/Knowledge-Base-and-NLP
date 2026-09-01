#!/bin/bash
dataset_name="msmarco_abs_500"

model_names=(
    "snowflake-llama-3.3-70b"
    "claude-3-5-sonnet"
    "deepseek-r1"
    "llama3.1-405b"
    "llama3.1-70b"
    "mistral-large"
    "mistral-large2"
    "mixtral-8x7b"
    "reka-flash"
    "snowflake-arctic"
)
search_cache_name="search_cache_abs_500.json"
top_k=10
max_tokens=32768
subset_num=10
context_length=2

for model_name in "${model_names[@]}"; do
    echo "======================================"
    echo "Start $model_name w/ $search_cache_name"
    CUDA_VISIBLE_DEVICES=6,7 \
    python scripts/cot_pilot_study.py \
        --dataset_name $dataset_name \
        --split test \
        --top_k $top_k \
        --max_tokens $max_tokens \
        --search_cache_name $search_cache_name \
        --model_path $model_name \
        --max_num_seqs 64 \
        --snowflake \
        --window_size 10 \
        --window_overlap 0 \
        --subset_num $subset_num \
        --context_length $context_length 
done
