# #!/bin/bash

export PYTHONPATH=$(pwd)
cuda_devices="0,1,2,3"

max_dlen=512
top_k_docs=1000
max_tokens=2048
prompt_mode=reasoning_and_ranking
model_name_or_path=/data/models/Qwen2.5-7B-Instruct
peft_model_name=./train/models/sft/v8_prefix_tevatron_72k_ndcg0.1_deepseek-r1/checkpoint-4500
graph_path=./data/wiki/wiki.contriever.16
corpus_path=./data/wiki/corpus.jsonl
d2d="contriever_td2d_freq2"


dataset_names=(
    "hotpotqa"
    "2wiki"
    "musique"
)

for dataset in "${dataset_names[@]}"; do
    CUDA_VISIBLE_DEVICES=$cuda_devices \
    CUDA_VISIBLE_DEVICES=$cuda_devices \
    python -u -m scripts.run \
        --task rerank \
        --window_size 20 \
        --batch_size 16 \
        --max_dlen ${max_dlen} \
        --top_k_docs ${top_k_docs} \
        --max_tokens ${max_tokens} \
        --prompt_mode ${prompt_mode} \
        --qa_path ../data/QA_Datasets/${dataset}/${dataset}_500.json \
        --cache_path ../cache/$dataset/search_cache_500_wiki_top1000.json \
        --model_name_or_path ${model_name_or_path} \
        --peft_model_name ${peft_model_name} \
        --gpu_memory_utilization 0.875 \
        --corpus_path $corpus_path \
        --graph_path $graph_path \
        --d2d ${d2d} \
        --return_trace \
        --vllm
done




