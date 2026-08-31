export PYTHONPATH='.'
export CUDA_VISIBLE_DEVICES=6,7

python llm/qgen.py --model-path "../../pretrained_models/Meta-Llama-3-8B-Instruct"