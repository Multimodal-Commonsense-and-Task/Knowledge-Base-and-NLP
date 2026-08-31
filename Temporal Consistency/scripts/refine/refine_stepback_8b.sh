export PYTHONPATH='.'
export CUDA_VISIBLE_DEVICES=4,5,6,7

python llm/timeqa_llama.py --model-path "../../pretrained_models/Meta-Llama-3-8B-Instruct" --do-refine \
        --output-path output/stepback/Meta-Llama-3-8B-Instruct/240701Jul-0_prompt_stepback_Meta-Llama-3-8B-Instruct.json


# --output-path output/output_prompt_stepback_Meta-Llama-3-8B-Instruct.json