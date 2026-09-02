# #!/bin/bash

export PYTHONPATH=$(pwd)
cuda_devices="4,5,6,7"


batch_size=1
max_dlen=512
gradient_accumulation_steps=8


CUDA_VISIBLE_DEVICES=$cuda_devices \
deepspeed --include localhost:$cuda_devices --master_port 60000 \
    scripts/run.py \
    --task train_sft \
    --batch_size $batch_size \
    --max_dlen $max_dlen \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --prompt_mode reasoning_and_ranking \
    --ds_config ./train/config/accelerate_configs/ds_zero0_config.json \
    --output_dir ./train/models/sft/v8_prefix_tevatron_60k_ndcg0.1_deepseek-r1 \
    --train_data_path ./train/data/train_sft_tevatron_60k_analysis_ndcg0.1_deepseek-r1.jsonl \
    --model_name_or_path /data/models/Qwen2.5-7B-Instruct




