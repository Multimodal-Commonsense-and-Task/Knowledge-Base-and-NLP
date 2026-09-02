#!/bin/bash


for dataset in "msmarco"
    do
        python -u calc_rerank_colbert_full.py --beir_dataset $dataset \
                                            --model_path v2/colbertv2.0 \
                                            --index_path index-colbertv2-len300 \
                                            --save_path /data/colbert-prf/analysis/hypo/hypo-colbertv2;
done