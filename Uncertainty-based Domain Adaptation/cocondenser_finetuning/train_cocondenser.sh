#!/bin/bash
set -e

train_path=$1
output_path=$2
batch_size=${3:-32}
save_steps=${4:-20000}
epochs=${5:-1}
run_name=${6:-"cocondenser-finetuning"}
lr=${7:-5e-6}
return_from=${8:-""}
model_name="Luyu/co-condenser-marco-retriever"

# if return_from is provided and not equal to empty string, resume from the checkpoint
if [ ! -z "${return_from}" ]; then
  echo "Resuming from checkpoint: ${return_from}"
  export WANDB_RESUME="must"
  python -m tevatron.driver.train \
    --output_dir ${output_path} \
    --model_name_or_path ${return_from} \
    --tokenizer_name ${model_name} \
    --save_steps ${save_steps} \
    --dataset_name Tevatron/msmarco-passage \
    --train_dir ${train_path} \
    --fp16 \
    --per_device_train_batch_size ${batch_size} \
    --learning_rate ${lr} \
    --num_train_epochs ${epochs} \
    --dataloader_num_workers 2 \
    --logging_steps 1
else
  python -m tevatron.driver.train \
    --output_dir ${output_path} \
    --model_name_or_path ${model_name} \
    --save_steps ${save_steps} \
    --dataset_name Tevatron/msmarco-passage \
    --train_dir ${train_path} \
    --fp16 \
    --per_device_train_batch_size ${batch_size} \
    --learning_rate ${lr} \
    --num_train_epochs ${epochs} \
    --dataloader_num_workers 2 \
    --logging_steps 1
fi
