#!/bin/bash

set -e

dir=$(dirname $0)
# Collection exists check
if [ ! -f "/your/path/beir-ColBERT/datasets/${1}-collection.tsv" ]; then
    echo "Preparing data for ${1}..."
    bash $dir/data_prep.sh $1
else
    echo "datasets/${1}-collection.tsv already exists"
fi

# Error ColBERT if not trained
if [ ! -d "/your/path/colbert_trained/${1}/${2}/MSMARCO-psg" ]; then
    echo "No Trained data of ColBERT for ${1}..."
    exit 1
else
    echo "ColBERT for ${1} already trained"
fi

# Reranking Test Data Gen of ColBERT
if [ ! -f "/your/path/colbert_trained/${1}/${2}/ranking-200/ranking.tsv" ]; then
    echo "Reranking Data Prepare ColBERT for ${1}..."
    bash $dir/top200.sh $1 $2
else
    echo "ColBERT for ${1} already have reranking data"
fi