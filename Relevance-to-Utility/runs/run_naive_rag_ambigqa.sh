#!/bin/bash

dataset_name="ambigqa"
input_file_name="runs.baselines"
ouptut_base_dir="runs.baselines"
subset_num=10000
max_tokens=20480
top_k=10
cuda_devices="0,1"

# Iterate through model names
model_names=(
  "/data/models/Llama-3.2-1B-Instruct" "/data/models/Llama-3.2-3B-Instruct" "/data/models/Llama-3.1-8B-Instruct" \
  "/data/models/Qwen2.5-0.5B-Instruct" "/data/models/Qwen2.5-1.5B-Instruct" "/data/models/Qwen2.5-3B-Instruct" "/data/models/Qwen2.5-7B-Instruct"
)

# Iterate through search cache names
search_cache_names=(
  "search_cache_500_compact.json" "search_cache_500_provence.json" \
  "search_cache_500_trained_rewriter_rewritten_llama-3.2-3b-comb-step-3480_split.json"
)

for model_name in "${model_names[@]}"; do
    for search_cache_name in "${search_cache_names[@]}"; do
        echo "======================================"
        echo "Start $model_name w/ $search_cache_name"

        CUDA_VISIBLE_DEVICES=$cuda_devices \
        python scripts/run_naive_rag.py \
            --dataset_name $dataset_name \
            --split test \
            --context_length 10 \
            --top_k $top_k \
            --max_tokens $max_tokens \
            --search_cache_name $search_cache_name \
            --model_path $model_name \
            --subset_num $subset_num
    done
done


# Iterate through model names
model_names=("/data/models/Qwen2.5-7B-Instruct")

# Iterate through search cache names
search_cache_names=(
  "search_cache_500.json" "search_cache_500_ranked.json" "search_cache_500_listwise_ranked.json" \
  "search_cache_500_compact.json" "search_cache_500_provence.json" \
  "search_cache_500_trained_rewriter_rewritten_llama-3.2-3b-comb-step-3480_split.json"
)

for model_name in "${model_names[@]}"; do
    for search_cache_name in "${search_cache_names[@]}"; do
        echo "======================================"
        echo "Start $model_name w/ $search_cache_name"

        CUDA_VISIBLE_DEVICES=$cuda_devices \
        python scripts/run_naive_rag.py \
            --dataset_name $dataset_name \
            --split test \
            --context_length 10 \
            --top_k $top_k \
            --max_tokens $max_tokens \
            --search_cache_name $search_cache_name \
            --model_path $model_name \
            --subset_num $subset_num
    done
done
