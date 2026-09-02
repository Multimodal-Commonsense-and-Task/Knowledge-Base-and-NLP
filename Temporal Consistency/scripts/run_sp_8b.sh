export PYTHONPATH='.'
export CUDA_VISIBLE_DEVICES=2,3

python timeqa_llama.py --icl 0 --model-path "../../pretrained_models/Meta-Llama-3-8B-Instructd"