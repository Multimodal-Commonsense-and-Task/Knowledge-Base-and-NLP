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

# Finetune ColBERT if not trained
if [ ! -d "/your/path/colbert_trained/${1}/${2}/MSMARCO-psg/train.pth" ]; then
    echo "Finetuning ColBERT for ${1}..."
    bash $dir/train_colbert.sh $1 $2
else
    echo "ColBERT for ${1} already trained"
fi

# Test ColBERT if not tested
if [ ! -f "/your/path/colbert_trained/${1}/${2}/ranking/metric.json" ]; then
    echo "Testing ColBERT for ${1}..."
    bash $dir/test_colbert.sh $1 $2
else
    echo "ColBERT for ${1} already tested"
fi