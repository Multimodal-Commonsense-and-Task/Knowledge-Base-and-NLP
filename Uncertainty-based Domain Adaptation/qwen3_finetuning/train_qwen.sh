#!/bin/bash
set -e

train_path=$1
output_path=$2
batch_size=${3:-32}
save_steps=${4:-20000}
epochs=${5:-1}
run_name=${6:-"qwen3-finetuning"}
lr=${7:-5e-6}
return_from=${8:-""}
dataset=${9:-""}
model_name="Qwen/Qwen3-Embedding-4B"

# prepare the query prefix according to the dataset
if [ -z "${dataset}" ]; then
  if [ ! -z "${return_from}" ]; then
    dataset=$return_from
    return_from=""
  fi
fi
if [ "$dataset" == "trec-news" ]; then
  query_prefix="Instruct: Classify the fine-grained category of the given news title.\nTitle:"
elif [ "$dataset" == "robust04" ]; then
  query_prefix="Instruct: Retrieval the relevant passage for the given query.\nQuery:"
elif [ "$dataset" == "trec-covid" ]; then
  query_prefix="Instruct: Given a query on COVID-19, retrieve documents that answer the query.\nQuery:"
elif [ "$dataset" == "quora" ]; then
  query_prefix="Instruct: Given a question, retrieve questions that are semantically equivalent to the given question.\nQuestion:"
elif [ "$dataset" == "hotpotqa" ]; then
  query_prefix="Instruct: Given a multi-hop question, retrieve documents that can help answer the question.\nQuestion:"
else
  echo "Unknown dataset: $dataset"
  exit 1
fi

export WANDB_PROJECT=${WANDB_PROJECT:-"unite"}
export WANDB_RUN_ID=$run_name
export WANDB_NAME=$run_name
# if return_from is provided and not equal to empty string, resume from the checkpoint
if [ ! -z "${return_from}" ]; then
  echo "Resuming from checkpoint: ${return_from}"
  # export WANDB_RESUME="must"
  deepspeed --include localhost:$CUDA_VISIBLE_DEVICES --master_port $(shuf -i 6000-7000 -n 1) --module tevatron.retriever.driver.train \
    --deepspeed deepspeed/ds_zero3_config.json \
    --output_dir ${output_path} \
    --model_name_or_path ${return_from} \
    --tokenizer_name ${model_name} \
    --lora \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
    --save_steps ${save_steps} \
    --dataset_path ${train_path} \
    --query_prefix "$query_prefix" \
    --passage_prefix "" \
    --bf16 \
    --pooling last \
    --padding_side left \
    --normalize \
    --temperature 0.01 \
    --per_device_train_batch_size ${batch_size} \
    --gradient_checkpointing \
    --train_group_size 16 \
    --learning_rate ${lr} \
    --num_train_epochs ${epochs} \
    --query_max_len 32 \
    --passage_max_len 128 \
    --dataloader_num_workers 2 \
    --logging_steps 1 \
    --overwrite_output_dir \
    --gradient_accumulation_steps 1 \
    --report_to wandb \
    --run_name $run_name
    # --resume_from_checkpoint ${return_from} \
else
    # --dataset_name Tevatron/msmarco-passage \
  deepspeed --include localhost:$CUDA_VISIBLE_DEVICES --master_port $(shuf -i 6000-7000 -n 1) --module tevatron.retriever.driver.train \
    --deepspeed deepspeed/ds_zero3_config.json \
    --output_dir ${output_path} \
    --model_name_or_path ${model_name} \
    --lora \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
    --save_steps ${save_steps} \
    --dataset_path ${train_path} \
    --query_prefix "$query_prefix" \
    --passage_prefix "" \
    --bf16 \
    --pooling last \
    --padding_side left \
    --normalize \
    --temperature 0.01 \
    --per_device_train_batch_size ${batch_size} \
    --gradient_checkpointing \
    --train_group_size 16 \
    --learning_rate ${lr} \
    --num_train_epochs ${epochs} \
    --query_max_len 32 \
    --passage_max_len 128 \
    --dataloader_num_workers 2 \
    --logging_steps 1 \
    --overwrite_output_dir \
    --gradient_accumulation_steps 1 \
    --report_to wandb \
    --run_name $run_name
fi
