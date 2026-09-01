#!/bin/bash
set -e

SCRIPT_DIR=$(dirname "$0")
dataset=$1
case=$2
force=${3:-false}
num_pos_to_neg=${4:-4}

# Generate hard negative training data if not exists
if [ -f "datasets/${dataset}_${case}_contriever_${num_pos_to_neg}_hardnegative_train_tevatron.jsonl" ]; then
    echo "datasets/${dataset}_${case}_contriever_${num_pos_to_neg}_hardnegative_train_tevatron.jsonl already exists"
    if [ "$force" = "false" ]; then
        echo "Exiting without generating hard negative training data."
        exit 0
    else
        echo "Force regenerating hard negative training data."
    fi
fi

query_path="datasets/${dataset}_query_${case}.jsonl"
echo "Generating hard negative training data for ${dataset}-${case}..."

python $SCRIPT_DIR/hardnegative_mining/train_data_generation.py \
    --dataset_name $dataset \
    --generated_queries_filepath $query_path \
    --intermediate_dir "temp/intermediate/${dataset}/${case}" \
    --save_reranker_traindata_filepath "datasets/${dataset}_${case}_contriever_${num_pos_to_neg}_hardnegative_train_reranker.jsonl" \
    --save_colbert_traindata_filepath "datasets/${dataset}_${case}_contriever_${num_pos_to_neg}_hardnegative_train_colbert.tsv" \
    --save_contriever_traindata_filepath "datasets/${dataset}_${case}_contriever_${num_pos_to_neg}_hardnegative_train_contriever.jsonl" \
    --save_tevatron_traindata_filepath "datasets/${dataset}_${case}_contriever_${num_pos_to_neg}_hardnegative_train_tevatron.jsonl" \
    --num_pos_to_neg $num_pos_to_neg