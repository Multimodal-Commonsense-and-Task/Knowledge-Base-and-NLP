export PYTHONPATH='.'
export CUDA_VISIBLE_DEVICES=0,1,2,3

python llm/timeqa_llama.py --prompt-type batch --model-path ../../pretrained_models/google-gemma-2-27b-it