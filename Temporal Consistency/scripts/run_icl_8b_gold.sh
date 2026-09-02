export PYTHONPATH='.'
export CUDA_VISIBLE_DEVICES=4,5

python llm/timeqa_llama.py --prompt-type icl --model-path "../../pretrained_models/Meta-Llama-3-8B-Instruct" --data-path dataset/test.hard.short.gold.json \
        --output-path output/short_gold.json