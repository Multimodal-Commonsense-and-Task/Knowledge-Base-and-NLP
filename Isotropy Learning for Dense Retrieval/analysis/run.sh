#!/bin/bash


for dataset in "nfcorpus" "scifact" "arguana" "scidocs" "fiqa" "trec-covid" "robust04" "trec-news" "touche" \
                "quora" "cqadupstack/android" "cqadupstack/english" "cqadupstack/gaming" "cqadupstack/gis" \
                "cqadupstack/mathematica" "cqadupstack/physics" "cqadupstack/programmers" "cqadupstack/stats" \
                "cqadupstack/tex" "cqadupstack/unix" "cqadupstack/webmasters" "cqadupstack/wordpress" \
                "nq" "hotpotqa" "signal1m" "dbpedia-entity" "fever" "climate-fever" "bioasq"
    do
        python -u calc_rerank_colbert_full.py --beir_dataset $dataset \
                                            --model_path v2/colbertv2.0 \
                                            --index_path index-colbertv2-len300 \
                                            --save_path /data/colbert-prf/analysis/hypo/hypo-colbertv2 \
                                            --bm25;
done
