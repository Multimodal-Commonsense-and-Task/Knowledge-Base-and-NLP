# #!/bin/bash

export PYTHONPATH=$(pwd)
cuda_devices="0,1,2,3"


max_dlen=512
top_k_docs=1000
max_tokens=2048
prompt_mode=reasoning_and_ranking
model_name_or_path=../../models/Qwen2.5-7B-Instruct
peft_model_name=./train/models/sft/v8_prefix_tevatron_72k_ndcg0.1_deepseek-r1/checkpoint-4500
d2d="contriever_td2d_freq2"

retriever=bm25

for beta in 0.0 0.2 0.4 0.6 0.8 1.0; do
    for alpha in 1.0 0.8 0.6 0.4 0.2 0.0; do
        for category in biology stackoverflow leetcode aops theoremqa_questions pony earth_science economics psychology robotics sustainable_living theoremqa_theorems; do
            CUDA_VISIBLE_DEVICES=${cuda_devices} \
            python -u -m scripts.run \
                --task rerank \
                --window_size 20 \
                --batch_size 48 \
                --max_dlen ${max_dlen} \
                --top_k_docs ${top_k_docs} \
                --max_tokens ${max_tokens} \
                --prompt_mode ${prompt_mode} \
                --qa_path ../data/QA_Datasets/bright/${category}.json \
                --cache_path ../cache/bright/${retriever}/search_cache_${retriever}_${category}.json \
                --model_name_or_path ${model_name_or_path} \
                --peft_model_name ${peft_model_name} \
                --gpu_memory_utilization 0.9 \
                --corpus_path ./data/bright/${category}/corpus.jsonl \
                --graph_path ./data/bright/${category}/${category}.contriever.32 \
                --d2d ${d2d} \
                --return_trace \
                --alpha ${alpha} \
                --beta ${beta} \
                --vllm
        done
    done
done

