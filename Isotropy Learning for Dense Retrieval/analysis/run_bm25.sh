#!/bin/bash

python -u calc_rerank_colbert.py --model_path msmarco-colbert \
                                --model_step 200000 \
                                --index_path index-colbert \
                                --save_path /data/colbert-prf/analysis/hypo/hypo-colbert-bm25-float32 \
                                --bm25 \
                                --all_dataset;