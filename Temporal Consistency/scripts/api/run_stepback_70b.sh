export PYTHONPATH='.'
export CUDA_VISIBLE_DEVICES=0,1,2,3,6,7

python llm/timeqa_llama_apicall.py --prompt-type stepback --model-path "../../pretrained_models/Meta-Llama-3-70B-Instruct" \
    --num-samples 100

python llm/timeqa_llama_apicall.py --prompt-type icl --model-path "../../pretrained_models/Meta-Llama-3-70B-Instruct" \
    --num-samples 100

# python llm/timeqa_llama_apicall.py --prompt-type sp --model-path "../../pretrained_models/Meta-Llama-3-70B-Instruct" \
#     --num-samples 100

