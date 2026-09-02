#!/bin/bash
SCRIPT_DIR=$(dirname "$0")
set -e

dataset=$1
python $SCRIPT_DIR/sampling/bm25_entropy_auto.py \
    --dataset_name $dataset \
    --save_entropy_dict_dir "intermediate/${dataset}" \
    --intermediate_dir "temp/${dataset}"
