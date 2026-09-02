#!/bin/bash

for dataset in "fiqa" "signal1m" "trec-news" \
                "robust04" "arguana" "touche" "quora" "dbpedia-entity" "scidocs" "fever" \
                "climate-fever" "scifact" "cqadupstack/android" "cqadupstack/english" \
                "cqadupstack/gaming" "cqadupstack/gis" "cqadupstack/mathematica" "cqadupstack/physics" \
                "cqadupstack/programmers" "cqadupstack/stats" "cqadupstack/tex" "cqadupstack/unix" \
                "cqadupstack/webmasters" "cqadupstack/wordpress"
    do
        python -u calc_rerank_colbert_full_ensemble.py --beir_dataset $dataset \
                                            --model_path msmarco-colbert \
                                            --model_type iso \
                                            --index_path index-colbert-seed24 \
                                            --save_path /data/colbert-prf/analysis/hypo/hypo-ensemble-variant-seed24;
done
