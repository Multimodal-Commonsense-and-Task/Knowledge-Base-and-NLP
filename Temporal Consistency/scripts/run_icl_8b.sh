export PYTHONPATH='.'
export CUDA_VISIBLE_DEVICES=0,1,2,3

python llm/timeqa_llama.py --prompt-type icl --model-path "../../pretrained_models/Meta-Llama-3-8B-Instruct"