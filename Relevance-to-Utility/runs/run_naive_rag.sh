# #!/bin/bash

dataset_name="hotpotqa"
input_file_name="runs.baselines"
ouptut_base_dir="runs.baselines"
subset_num=10000
max_tokens=20480
top_k=10
cuda_devices="0,1,2,3"

# Iterate through model names
model_names=(
  "/data/models/Qwen2.5-3B-Instruct"
)

# Iterate through search cache names
search_cache_names=(
  "search_cache_500.json"
)

for model_name in "${model_names[@]}"; do
    for search_cache_name in "${search_cache_names[@]}"; do
        echo "======================================"
        echo "Start $model_name w/ $search_cache_name"

        CUDA_VISIBLE_DEVICES=$cuda_devices \
        python scripts/run_naive_rag.py \
            --dataset_name $dataset_name \
            --split test \
            --top_k $top_k \
            --max_tokens $max_tokens \
            --search_cache_name $search_cache_name \
            --model_path $model_name \
            --subset_num $subset_num
    done
done
