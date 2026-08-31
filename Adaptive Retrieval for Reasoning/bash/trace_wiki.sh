# #!/bin/bash

cuda_devices="4"
prompt_mode=reasoning_and_ranking
max_tokens=768

# model_name_or_path=/data/models/Llama-3.1-8B-Instruct
# model_name_or_path=snowflake-llama-3.3-70b
model_name_or_path=deepseek-r1

dataset_names=(
    "train"
)

for dataset_name in "${dataset_names[@]}"; do
    CUDA_VISIBLE_DEVICES=$cuda_devices \
    python -u -m scripts.run \
        --task generate_data \
        --window_size 20 \
        --batch_size 64 \
        --top_k_docs 20 \
        --max_tokens $max_tokens \
        --prompt_mode $prompt_mode \
        --cache_path ../cache/msmarco_passage/search_cache_train_tevatron_full_72k_1.json \
        --model_name_or_path $model_name_or_path \
        --snowflake
done