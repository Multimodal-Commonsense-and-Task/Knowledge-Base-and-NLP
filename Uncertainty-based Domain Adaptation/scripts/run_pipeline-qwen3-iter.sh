#!/bin/bash
set -e
SCRIPT_DIR=$(dirname "$0")

dataset=$1
entropy=$2
iter=${3:-0}
lr=${4:-5e-6}
contamination=${5:-1.5}
n_train_per_iter=500

save_step=1000
cluster=1000
logit_k=1000
model="qwen3"
version="online${iter}"
num_pos_to_neg=4
case="${model}_${version}"
run_name="${model}_${dataset}_online"

echo "Running Qwen3 pipeline for dataset: $dataset, iteration: $iter"

$SCRIPT_DIR/build_idf.sh "$dataset" "Qwen/Qwen3-Embedding-4B" true "$entropy" "$contamination"
if [ $iter != 0 ]; then
    prev_version="online$((iter-1))"
    prev_case="${model}_${prev_version}"
    checkpoint_dir=$(ls -d qwen3_trained/${dataset}/${prev_case}_contriever_4_seed42_tevatron | sort -V | tail -n 1)
    last_checkpoint=$(ls -d ${checkpoint_dir}/checkpoint-* | sort -V | tail -n 1)
    $SCRIPT_DIR/encode_trained.sh "$dataset" "$model" "cls" "$checkpoint_dir"
    $SCRIPT_DIR/mlm_idf_score_embedding.sh "$dataset" "$model" "$version" "cls" "${checkpoint_dir}/embeddings_beir"
    if $SCRIPT_DIR/early_stop.sh "$dataset" "$model" "online"; then
        echo "Early stop detected - stopping training"
        exit 1
    else
        echo "No early stop - continuing training"
    fi
    $SCRIPT_DIR/sample_embedding_rm_prev_penalty.sh "$dataset" "$case" "$model" "$version" "$logit_k" "$cluster" "$n_train_per_iter" "$entropy" "$contamination" "cls" "${checkpoint_dir}/embeddings_beir"
    $SCRIPT_DIR/qgen.sh "$dataset" "$case"
    $SCRIPT_DIR/hard_negative.sh "$dataset" "$case" false "$num_pos_to_neg"
    $SCRIPT_DIR/run_epochs-qwen3.sh "$dataset" "${case}_contriever_${num_pos_to_neg}" 1 "$save_step" 4 "$run_name" "$lr" "$last_checkpoint"
else
    $SCRIPT_DIR/encode.sh "$dataset" "$model" "cls"
    $SCRIPT_DIR/mlm_idf_score.sh "$dataset" "$model" "$version" "cls"
    $SCRIPT_DIR/sample.sh "$dataset" "$case" "$model" "$version" "$logit_k" "$cluster" "$n_train_per_iter" "$entropy" "$contamination"
    $SCRIPT_DIR/qgen.sh "$dataset" "$case"
    $SCRIPT_DIR/hard_negative.sh "$dataset" "$case" false "$num_pos_to_neg"
    $SCRIPT_DIR/run_epochs-qwen3.sh "$dataset" "${case}_contriever_${num_pos_to_neg}" 1 "$save_step" 4 "$run_name" "$lr"
fi
