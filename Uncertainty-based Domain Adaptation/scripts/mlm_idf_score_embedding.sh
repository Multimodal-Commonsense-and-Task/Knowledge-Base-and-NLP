#!/bin/bash
SCRIPT_DIR=$(dirname "$0")
dataset=$1
model=$2
version=$3
pooling=${4:-"cls"}
embedding_path=$5
suffix="_nostop_aleatoric_z_sigma"

# check model name is colbert

if [[ $model == "colbert" ]]; then
    echo "Model is colbert"
    if [ ! -f "intermediates/${dataset}/${model}/${version}/logit_1000.pt" ]; then
        echo "MLM score file does not exist, generating..."
        python $SCRIPT_DIR/sampling/document_score_token.py \
            --dataset $dataset \
            --idf_dict_path "intermediates/${dataset}/idf_dict_bert-base-uncased${suffix}.json" \
            --document_embedding_path $embedding_path \
            --collection_path "intermediates/${dataset}/colbert/collection.tsv" \
            --embedding_model $model \
            --cleaned_document_path "intermediates/${dataset}/cleaned_documents.jsonl" \
            --score_dir "intermediates/${dataset}/${model}/${version}" \
            --output_file "intermediates/${dataset}/${model}/${version}/output.pt"
    else
        echo "MLM score file already exists"
    fi
elif [[ $model == "monot5" ]]; then
    echo "Model is monot5"
    if [ ! -f "intermediates/${dataset}/${model}/${version}/logit_1000.pt" ]; then
        echo "MLM score file does not exist, generating..."
        python $SCRIPT_DIR/sampling/document_score_token.py \
            --dataset $dataset \
            --idf_dict_path "intermediates/${dataset}/idf_dict_t5-base${suffix}.json" \
            --document_embedding_path $embedding_path \
            --embedding_model $model \
            --cleaned_document_path "intermediates/${dataset}/cleaned_documents.jsonl" \
            --output_file "intermediates/${dataset}/${model}/${version}/output.pt" \
            --score_dir "intermediates/${dataset}/${model}/${version}"
    else
        echo "MLM score file already exists"
    fi
elif [[ $model == "dpr" ]]; then
    echo "Model is dpr"
    if [ ! -f "intermediates/${dataset}/${model}/${version}/logit_1000.pt" ]; then
        echo "MLM score file does not exist, generating..."
        python $SCRIPT_DIR/sampling/document_score_token.py \
            --dataset $dataset \
            --idf_dict_path "intermediates/${dataset}/idf_dict_bert-base-uncased${suffix}.json" \
            --document_embedding_path $embedding_path \
            --embedding_model $model \
            --cleaned_document_path "intermediates/${dataset}/cleaned_documents.jsonl" \
            --output_file "intermediates/${dataset}/${model}/${version}/output.pt" \
            --score_dir "intermediates/${dataset}/${model}/${version}"
    else
        echo "MLM score file already exists"
    fi
elif [[ $model == "cocondenser" ]]; then
    echo "Model is cocondenser"
    if [ ! -f "intermediates/${dataset}/${model}/${version}/logit_1000.pt" ]; then
        echo "MLM score file does not exist, generating..."
        python $SCRIPT_DIR/sampling/document_score_token.py \
            --dataset $dataset \
            --idf_dict_path "intermediates/${dataset}/idf_dict_bert-base-uncased${suffix}.json" \
            --document_embedding_path $embedding_path \
            --embedding_model $model \
            --cleaned_document_path "intermediates/${dataset}/cleaned_documents.jsonl" \
            --output_file "intermediates/${dataset}/${model}/${version}/output.pt" \
            --score_dir "intermediates/${dataset}/${model}/${version}"
    else
        echo "MLM score file already exists"
    fi
elif [[ $model == "cocodr" ]]; then
    echo "Model is cocodr"
    if [ ! -f "intermediates/${dataset}/${model}/${version}/logit_1000.pt" ]; then
        echo "MLM score file does not exist, generating..."
        python $SCRIPT_DIR/sampling/document_score_token.py \
            --dataset $dataset \
            --idf_dict_path "intermediates/${dataset}/idf_dict_bert-base-uncased${suffix}.json" \
            --document_embedding_path $embedding_path \
            --embedding_model $model \
            --cleaned_document_path "intermediates/${dataset}/cleaned_documents.jsonl" \
            --output_file "intermediates/${dataset}/${model}/${version}/output.pt" \
            --score_dir "intermediates/${dataset}/${model}/${version}"
    else
        echo "MLM score file already exists"
    fi

elif [[ $model == "qwen3" ]]; then
    echo "Model is qwen3"
    if [ ! -f "intermediates/${dataset}/${model}/${version}/logit_1000.pt" ]; then
        echo "MLM score file does not exist, generating..."
        python $SCRIPT_DIR/sampling/document_score_token.py \
            --dataset $dataset \
            --idf_dict_path "intermediates/${dataset}/idf_dict_Qwen-Qwen3-Embedding-4B${suffix}.json" \
            --document_embedding_path $embedding_path \
            --embedding_model $model \
            --mlm_model "Qwen/Qwen3-4B" \
            --cleaned_document_path "intermediates/${dataset}/cleaned_documents.jsonl" \
            --output_file "intermediates/${dataset}/${model}/${version}/output.pt" \
            --score_dir "intermediates/${dataset}/${model}/${version}"
    else
        echo "MLM score file already exists"
    fi
else
    exit 1
fi