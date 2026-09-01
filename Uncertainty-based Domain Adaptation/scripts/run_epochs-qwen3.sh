#!/bin/bash
set -e

dataset=$1
case=$2
epochs=${3:-1}
save_steps=${4:-1000}
batch_size=${5:-4}
run_name=${6:-qwen3_${dataset}_${case}}
lr=${7:-5e-6}
last_checkpoint=${8:-""}
seed=42

trained_dir="qwen3_trained/${dataset}/${case}_seed${seed}_tevatron"

echo "Running for dataset: $dataset, case: $case, epochs: $epochs, save_steps: $save_steps, seed: $seed"
if [ ! -d "$trained_dir" ]; then
    echo "Finetuning qwen3 for ${dataset}..."
    train_path="datasets/${dataset}_${case}_hardnegative_train_tevatron.jsonl"
    qwen3_finetuning/train_qwen.sh "$train_path" "$trained_dir" "$batch_size" "$save_steps" "$epochs" "$run_name" "$lr" "$last_checkpoint" "$dataset"
fi

if [ -d "$trained_dir" ] && [ ! -f "${trained_dir}/embeddings_beir/metrics.json" ]; then
    echo "Evaluating qwen3 for ${dataset}... ${case}-${trained_dir}"
    qwen3_finetuning/test_qwen.sh "$trained_dir" "$dataset" "$run_name"
fi
