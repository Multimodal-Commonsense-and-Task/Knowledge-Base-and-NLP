# #!/bin/bash

export PYTHONPATH=$(pwd)
cuda_devices="0,1,2,3"

max_dlen=512
max_tokens=2048
max_sunar_tokens=256
prompt_mode=pointwise
model_name_or_path=/data/models/Llama-3.1-8B-Instruct
d2d=contriever_d2d_freq2


graph_path=./data/wiki/wiki.contriever.16
corpus_path=./data/wiki/corpus.jsonl
dataset_names=(
    "hotpotqa"
    "2wiki"
    "musique"
)
for dataset in "${dataset_names[@]}"; do
    CUDA_VISIBLE_DEVICES=$cuda_devices \
    python -u -m scripts.run \
        --task sunar \
        --window_size 20 \
        --batch_size 16 \
        --max_dlen $max_dlen \
        --max_tokens $max_tokens \
        --max_sunar_tokens $max_sunar_tokens \
        --prompt_mode $prompt_mode \
        --qa_path ../data/QA_Datasets/$dataset/$dataset.json \
        --cache_path ../cache/$dataset/search_cache_500_wiki_top1000.json \
        --model_name_or_path $model_name_or_path \
        --gpu_memory_utilization 0.9 \
        --corpus_path $corpus_path \
        --graph_path $graph_path \
        --d2d $d2d \
        --vllm
done

