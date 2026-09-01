#!/bin/bash
SCRIPT_DIR=$(dirname "$0")
dataset=$1
case=$2
model=$3
version=$4
logit_k=$5
clusters=${6:-1000}
n_train=${7:-20000}
entropy=${8:-10}
contamination=${9:-0.05}
pooling=${10:-"cls"}
embedding_path=${11}
duqgen_filter=${12:-false}
alpha=${13:-0.5}

# Skip if the file already exists
if [ -f "datasets/${dataset}_sampled_${case}.jsonl" ]; then
    echo "Sampled data already exists for ${dataset}-${case}. Skipping..."
    exit 0
fi

if [[ $model == "colbert" ]]; then
    echo "Model is colbert"
    python $SCRIPT_DIR/sampling/epistemic_sampling_rm_prev_penalty.py \
        --dataset_name $dataset \
        --cleaned_document_path "intermediates/${dataset}/cleaned_documents.jsonl" \
        --document_embedding_path $embedding_path \
        --entropy_dict_path "intermediates/${dataset}/klentropy_k${entropy}.json" \
        --mlm_scores_path "intermediates/${dataset}/${model}/${version}/logit_${logit_k}.pt" \
        --collection_path "intermediates/${dataset}/colbert/collection.tsv" \
        --model_name "colbert" \
        --save_sampled_documents_filepath "datasets/${dataset}_sampled_${case}.jsonl" \
        --sampling_method "clustering" \
        --n_clusters $clusters \
        --n_train $n_train \
        --remove_outlier "entropy" \
        --alpha $alpha \
        --duqgen_filter $duqgen_filter \
        --contamination $contamination
elif [[ $model == "monot5" ]]; then
    echo "Model is monot5"
    python $SCRIPT_DIR/sampling/epistemic_sampling_rm_prev_penalty.py \
        --dataset_name $dataset \
        --cleaned_document_path "intermediates/${dataset}/cleaned_documents.jsonl" \
        --document_embedding_path $embedding_path \
        --entropy_dict_path "intermediates/${dataset}/klentropy_k${entropy}.json" \
        --mlm_scores_path "intermediates/${dataset}/${model}/${version}/logit_${logit_k}.pt" \
        --model_name "monot5" \
        --save_sampled_documents_filepath "datasets/${dataset}_sampled_${case}.jsonl" \
        --sampling_method "clustering" \
        --n_clusters $clusters \
        --n_train $n_train \
        --remove_outlier "entropy" \
        --alpha $alpha \
        --duqgen_filter $duqgen_filter \
        --contamination $contamination
elif [[ $model == "dpr" ]]; then
    echo "Model is dpr"
    python $SCRIPT_DIR/sampling/epistemic_sampling_rm_prev_penalty.py \
        --dataset_name $dataset \
        --cleaned_document_path "intermediates/${dataset}/cleaned_documents.jsonl" \
        --document_embedding_path $embedding_path \
        --entropy_dict_path "intermediates/${dataset}/klentropy_k${entropy}.json" \
        --mlm_scores_path "intermediates/${dataset}/${model}/${version}/logit_${logit_k}.pt" \
        --model_name "dpr" \
        --save_sampled_documents_filepath "datasets/${dataset}_sampled_${case}.jsonl" \
        --sampling_method "clustering" \
        --n_clusters $clusters \
        --n_train $n_train \
        --remove_outlier "entropy" \
        --alpha $alpha \
        --duqgen_filter $duqgen_filter \
        --contamination $contamination
elif [[ $model == "cocondenser" ]]; then
    echo "Model is cocondenser"
    python $SCRIPT_DIR/sampling/epistemic_sampling_rm_prev_penalty.py \
        --dataset_name $dataset \
        --cleaned_document_path "intermediates/${dataset}/cleaned_documents.jsonl" \
        --document_embedding_path $embedding_path \
        --entropy_dict_path "intermediates/${dataset}/klentropy_k${entropy}.json" \
        --mlm_scores_path "intermediates/${dataset}/${model}/${version}/logit_${logit_k}.pt" \
        --model_name "cocondenser" \
        --save_sampled_documents_filepath "datasets/${dataset}_sampled_${case}.jsonl" \
        --sampling_method "clustering" \
        --n_clusters $clusters \
        --n_train $n_train \
        --remove_outlier "entropy" \
        --alpha $alpha \
        --duqgen_filter $duqgen_filter \
        --contamination $contamination
elif [[ $model == "cocodr" ]]; then
    echo "Model is cocodr"
    python $SCRIPT_DIR/sampling/epistemic_sampling_rm_prev_penalty.py \
        --dataset_name $dataset \
        --cleaned_document_path "intermediates/${dataset}/cleaned_documents.jsonl" \
        --document_embedding_path $embedding_path \
        --entropy_dict_path "intermediates/${dataset}/klentropy_k${entropy}.json" \
        --mlm_scores_path "intermediates/${dataset}/${model}/${version}/logit_${logit_k}.pt" \
        --model_name "cocodr" \
        --save_sampled_documents_filepath "datasets/${dataset}_sampled_${case}.jsonl" \
        --sampling_method "clustering" \
        --n_clusters $clusters \
        --n_train $n_train \
        --remove_outlier "entropy" \
        --alpha $alpha \
        --duqgen_filter $duqgen_filter \
        --contamination $contamination

elif [[ $model == "qwen3" ]]; then
    echo "Model is qwen3"
    python $SCRIPT_DIR/sampling/epistemic_sampling_rm_prev_penalty.py \
        --dataset_name $dataset \
        --cleaned_document_path "intermediates/${dataset}/cleaned_documents.jsonl" \
        --document_embedding_path $embedding_path \
        --entropy_dict_path "intermediates/${dataset}/klentropy_k${entropy}.json" \
        --mlm_scores_path "intermediates/${dataset}/${model}/${version}/logit_${logit_k}.pt" \
        --model_name "qwen3" \
        --save_sampled_documents_filepath "datasets/${dataset}_sampled_${case}.jsonl" \
        --sampling_method "clustering" \
        --n_clusters $clusters \
        --n_train $n_train \
        --remove_outlier "entropy" \
        --alpha $alpha \
        --duqgen_filter $duqgen_filter \
        --contamination $contamination
else
    exit 1
fi