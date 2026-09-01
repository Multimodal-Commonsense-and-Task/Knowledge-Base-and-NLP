#!/bin/bash
SCRIPT_DIR=$(dirname "$0")
dataset=$1
model=$2
model_cleaned=${model//\//-}
entropy=${4:-3}
contamination=${5:-1.5}


echo "Stopword filter is enabled."
idf_dict_file="intermediates/${dataset}/idf_dict_${model_cleaned}_nostop_aleatoric_z_sigma.json"
corpus_tokenized_file="intermediates/${dataset}/corpus_tokenized_${model_cleaned}_nostop.pt"

# skip if the all file exists
if [ -f $corpus_tokenized_file ]; then
    if [ -f $idf_dict_file ]; then
        echo "IDF dict already exists for ${dataset}. Skipping..."
        exit 0
    fi
fi

python $SCRIPT_DIR/target_representation/generate_idf_dictionary_with_config_aleatoric_z_sigma.py \
    --dataset $dataset \
    --embedding_model $model \
    --clean_document_path "intermediates/${dataset}/cleaned_documents.jsonl" \
    --corpus_idf_dict_file $idf_dict_file \
    --corpus_tokenized_file $corpus_tokenized_file \
    --entropy_dict_path "intermediates/${dataset}/klentropy_k${entropy}.json" \
    --contamination $contamination \
    --stopword_filter