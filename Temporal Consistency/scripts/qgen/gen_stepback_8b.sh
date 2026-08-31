export PYTHONPATH='.'
export CUDA_VISIBLE_DEVICES=4,5

python llm/qgen_llm/qgen.py --model-path "../../pretrained_models/Meta-Llama-3-8B-Instruct" --prompt-type "stepback"