#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
python pruner/provence.py \
  --dataset_name 'ambigqa' \
  --input_file search_cache_500.json