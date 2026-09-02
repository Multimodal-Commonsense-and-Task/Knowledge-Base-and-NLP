export PYTHONPATH='.'
export CUDA_VISIBLE_DEVICES=6,7

python llm/timeqa_llama.py --model-path "../../pretrained_models/Meta-Llama-3-8B-Instruct" --do-refine \
        --output-path output/2024-06-30/output_prompt_icl_Meta-Llama-3-8B-Instruct.json --prompt-type batch