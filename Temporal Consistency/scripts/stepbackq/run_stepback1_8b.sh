export PYTHONPATH='.'
export CUDA_VISIBLE_DEVICES=6,7

python llm/timeqa_llama.py --prompt-type stepback1 --model-path "../../pretrained_models/Meta-Llama-3-8B-Instruct" --output-path output/abstract/abstract.jsonl --data-path dataset/test.stepback_question.json