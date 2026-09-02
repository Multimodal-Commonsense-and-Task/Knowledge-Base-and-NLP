#!/bin/bash
REPO_DIR="/data/FIRST"
# Define model, dataset paths, and output directory
MODEL_NAME="Llama-3.2-3B-Instruct"
BASE_MODEL="/data/models/$MODEL_NAME"
TRAIN_DATA_PATH="datasets/msmarco_train.jsonl"  # Train Dataset --> Hugging Face dataset or Local dataset
EVAL_DATA_PATH="datasets/msmarco_dev.jsonl"  # Eval Dataset --> Hugging Face dataset or Local dataset
OUTPUT_DIR="${REPO_DIR}/models/rewriting/$MODEL_NAME"  # Directory to save the trained model
BEIR_DATA_DIR="${REPO_DIR}/datasets/beir/"


# Launch training with DeepSpeed configuration
export CUDA_VISIBLE_DEVICES=4,5,6,7
accelerate launch --config_file "${REPO_DIR}/train_configs/accel_config_deepspeed.yaml" "${REPO_DIR}/scripts/train_rewriting.py" \
    --model_name_or_path "${BASE_MODEL}" \
    --train_dataset_path "${TRAIN_DATA_PATH}" \
    --eval_dataset_path "${EVAL_DATA_PATH}" \
    --beir_data_path "${BEIR_DATA_DIR}" \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 3 \
    --seed 42 \
    --per_device_train_batch_size 2 \
    --eval_steps 1000 \
    --checkpointing_steps 870 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 50 \
    --output_dir "${OUTPUT_DIR}" \
    --noisy_embedding_alpha 5 \
    --objective generation \
    --report_to tensorboard \
    --with_tracking