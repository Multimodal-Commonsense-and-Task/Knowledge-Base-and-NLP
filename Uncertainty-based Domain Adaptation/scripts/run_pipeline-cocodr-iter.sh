#!/bin/bash
SCRIPT_DIR=$(dirname "$0")
set -e
dataset=$1
entropy=$2
iter=${3:-0}
lr=${4:-5e-6}
contamination=${5:-1.5}
n_train_per_iter=500


save_step=1000
cluster=1000
logit_k=1000
model="cocodr"
version="online${iter}"
echo "Running pipeline for dataset: $dataset"
num_pos_to_neg=4
case="${model}_${version}"
run_name=$case


# Run the encode script
$SCRIPT_DIR/scripts/build_idf_aleatoric_z_sigma.sh $dataset "bert-base-uncased" true $entropy $contamination
if [ $iter != 0 ]; then
    echo "This is iteration $iter"
    prev_version="online$((iter-1))"
    prev_case="${model}_${prev_version}"
    checkpoint_dir=$(ls -d cocodr_trained/${dataset}/${prev_case}_contriever_4_seed42_tevatron | sort -V | tail -n 1)
    last_checkpoint=$(ls -d ${checkpoint_dir}/checkpoint-* | sort -V | tail -n 1)
    $SCRIPT_DIR/scripts/encode_trained.sh $dataset $model "cls" $checkpoint_dir
    $SCRIPT_DIR/scripts/mlm_idf_score_embedding.sh $dataset $model $version false "cls" "${checkpoint_dir}/embeddings_beir"
    if $SCRIPT_DIR/scripts/early_stop.sh $dataset $model "online"; then
        echo "Early stop detected - stopping training"
        exit 1
    else
        echo "No early stop - continuing training"
    fi
    $SCRIPT_DIR/scripts/sample_embedding_rm_prev_penalty.sh $dataset $case $model $version $logit_k $cluster $n_train_per_iter $entropy $contamination "cls" "${checkpoint_dir}/embeddings_beir"
    $SCRIPT_DIR/scripts/qgen.sh $dataset $case
    $SCRIPT_DIR/scripts/hard_negative.sh $dataset $case false $num_pos_to_neg
    $SCRIPT_DIR/scripts/run_epochs-cocodr.sh $dataset "${case}_contriever_${num_pos_to_neg}" 1 $save_step 32 $run_name $lr $last_checkpoint
else
    echo "This is iteration 0"
    $SCRIPT_DIR/scripts/encode.sh $dataset $model "cls"
    $SCRIPT_DIR/scripts/mlm_idf_score.sh $dataset $model $version false
    $SCRIPT_DIR/scripts/sample.sh $dataset $case $model $version $logit_k $cluster $n_train_per_iter $entropy $contamination
    $SCRIPT_DIR/scripts/qgen.sh $dataset $case
    $SCRIPT_DIR/scripts/hard_negative.sh $dataset $case false $num_pos_to_neg
    $SCRIPT_DIR/scripts/run_epochs-cocodr.sh $dataset "${case}_contriever_${num_pos_to_neg}" 1 $save_step 32 $run_name $lr
fi