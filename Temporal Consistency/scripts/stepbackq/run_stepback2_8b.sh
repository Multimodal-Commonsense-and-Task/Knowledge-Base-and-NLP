export PYTHONPATH='.'
export CUDA_VISIBLE_DEVICES=4,5

python llm/timeqa_llama.py --prompt-type stepback2 --model-path "../../pretrained_models/Meta-Llama-3-8B-Instruct" --output-path output/abstract/abstract.jsonl --data-path dataset/test.stepback_question.json