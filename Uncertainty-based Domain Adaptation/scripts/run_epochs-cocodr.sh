#!/bin/bash
dataset=$1
case=$2
epochs=${3:-1}
save_steps=${4:-1000}
batch_size=${5:-32}
run_name=${6}
lr=${7:-5e-6}
last_checkpoint=${8:-""}
seed=42

echo "Running for dataset: $dataset, case: $case, epochs: $epochs, save_steps: $save_steps, seed: $seed"
if [ ! -d "cocodr_trained/${dataset}/${case}_seed${seed}_tevatron" ]; then
    echo "Finetuning cocodr for ${dataset}..."
    train_path="datasets/${dataset}_${case}_hardnegative_train_tevatron.jsonl"
    output_path="cocodr_trained/${dataset}/${case}_seed${seed}_tevatron"
    cocodr_finetuning/train_cocodr.sh $train_path $output_path $batch_size $save_steps $epochs $run_name $lr $last_checkpoint
fi

# if directory name is not encodes scores
for subdir in $(echo "cocodr_trained/${dataset}/${case}_seed${seed}_tevatron" | tr ' ' '\n' | sort -V | tr '\n' ' '); do
    if [ -d "$subdir" ]; then
        if [[ $subdir != *"encodes"* && $subdir != *"scores"* ]]; then 
            if [ ! -d "${subdir}/evaluation_results" ]; then
                echo "Evaluating cocodr for ${dataset}... ${case}-${subdir}"
                cocodr_finetuning/test_cocodr.sh $subdir $dataset $run_name
            fi
        fi
    fi
done