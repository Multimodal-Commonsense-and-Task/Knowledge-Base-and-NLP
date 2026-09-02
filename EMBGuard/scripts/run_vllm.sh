#!/bin/bash

# vLLM server startup script
# Usage: bash scripts/run_vllm.sh [max_model_len]

# ============================================
# Hugging Face Token Configuration
# ============================================
# Option 1: Set environment variable before running: export HF_TOKEN="your_token"
# Option 2: Set it here directly (uncomment and edit):
HF_TOKEN="xxx"

if [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN
    export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
    echo "HF_TOKEN is set"
else
    echo "Warning: HF_TOKEN is not set. You may not be able to download private models from Hugging Face Hub."
fi

# Allow long max model length
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Default configuration
# Note: Qwen3-VL does not support tensor parallel, so TENSOR_PARALLEL_SIZE must be 1
TENSOR_PARALLEL_SIZE=1
CUDA_VISIBLE_DEVICES=4
MODEL_NAME_OR_PATH="EMBGuard/EMBGuard-2B"
# MODEL_NAME_OR_PATH="EMBGuard/EMBGuard-4B"
HOST=127.0.0.1
PORT=8005
MAX_MODEL_LEN=18000  # Default to 8192 if not provided

# Run vLLM server
# Note: Qwen3-VL has known issues with vLLM v1 engine's multimodal handling
# Using legacy engine via VLLM_USE_V1=0 environment variable
# Disable flash-attention to avoid compatibility issues
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME_OR_PATH \
    --tokenizer $MODEL_NAME_OR_PATH \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --trust-remote-code \
    --max_num_seqs 4 \
    --seed 42 \
    --gpu-memory-utilization 0.9 \
    --enforce-eager \
    --host $HOST \
    --port $PORT \
    --max-model-len $MAX_MODEL_LEN 


