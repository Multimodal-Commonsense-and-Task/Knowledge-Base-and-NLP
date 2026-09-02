#!/bin/bash
SCRIPT_DIR=$(dirname "$0")
set -e
dataset=$1
entropy=$2
iter=${3:-0}
contamination=${4:-1.5}
n_train_per_iter=500

echo "Running pipeline for dataset: $dataset"
version="online${iter}"
model="dpr"
num_pos_to_neg=4
save_step=1000
cluster=1000
logit_k=1000
case="${model}_${version}"

$SCRIPT_DIR/scripts/build_idf.sh $dataset "bert-base-uncased" true $entropy $contamination
if [ $iter != 0 ]; then
    prev_version="online$((iter-1))"
    prev_case="${model}_${prev_version}"
    checkpoint_dir=$(ls -d dpr_trained/${dataset}/${prev_case}_contriever_4_seed42/*/ | grep checkpoint_ | sort -V | tail -n 1)
    $SCRIPT_DIR/scripts/encode_trained.sh $dataset $model "cls" $checkpoint_dir
    $SCRIPT_DIR/scripts/mlm_idf_score_embedding.sh $dataset $model $version "cls" "${checkpoint_dir}/embeddings.pt"
    if $SCRIPT_DIR/scripts/early_stop.sh $dataset $model "online"; then
        echo "Early stop detected - stopping training"
        exit 1
    else
        echo "No early stop - continuing training"
    fi
    $SCRIPT_DIR/scripts/sample_embedding_rm_prev_penalty.sh $dataset $case $model $version $logit_k $cluster $n_train_per_iter $entropy $contamination "cls" "${checkpoint_dir}/embeddings.pt"
    $SCRIPT_DIR/scripts/qgen.sh $dataset $case
    $SCRIPT_DIR/scripts/hard_negative.sh $dataset $case false $num_pos_to_neg
    $SCRIPT_DIR/scripts/run_epochs-dpr-checkpoint.sh $dataset "${case}_contriever_${num_pos_to_neg}" 1 $save_step true $checkpoint_dir
else
    $SCRIPT_DIR/scripts/encode.sh $dataset $model "cls"
    $SCRIPT_DIR/scripts/mlm_idf_score.sh $dataset $model $version "cls"
    $SCRIPT_DIR/scripts/sample.sh $dataset $case $model $version $logit_k $cluster $n_train_per_iter $entropy $contamination
    $SCRIPT_DIR/scripts/qgen.sh $dataset $case
    $SCRIPT_DIR/scripts/hard_negative.sh $dataset $case false $num_pos_to_neg
    $SCRIPT_DIR/scripts/run_epochs-dpr.sh $dataset "${case}_contriever_${num_pos_to_neg}" 1 $save_step true
fi