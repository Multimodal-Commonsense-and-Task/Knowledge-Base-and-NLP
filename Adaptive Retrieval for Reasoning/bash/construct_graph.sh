# #!/bin/bash

export PYTHONPATH=$(pwd)
cuda_devices="4"


categories=(
    "aops"
)

for category in "${categories[@]}"; do
    mkdir -p ./data/bright/$category

    CUDA_VISIBLE_DEVICES=$cuda_devices \
    python -u -m scripts.run \
        --task construct_graph \
        --model_name_or_path facebook/contriever-msmarco \
        --qa_path ../data/QA_Datasets/bright/${category}.json \
        --corpus_path ./data/bright/$category/corpus.jsonl \
        --output_dir ./data/bright/$category \
        --dataset $category \
        --k 11250 \
        --batch_size 128
done
