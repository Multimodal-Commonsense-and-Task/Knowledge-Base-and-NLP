#!/bin/bash
dataset=$1
case=$2
epochs=${3:-1}
save_steps=${4:-1000}
eval_last_only=${5:-true}
checkpoint=$6
seed=42

echo "Running for dataset: $dataset, case: $case, epochs: $epochs, save_steps: $save_steps, seed: $seed"
if [ ! -d "dpr_trained/${dataset}/${case}_seed${seed}" ]; then
    echo "Finetuning dpr for ${dataset}..."
    python dpr_finetuning/train_dpr_continue.py \
        --base_modename_or_path $checkpoint \
        --triplets_path "datasets/${dataset}_${case}_hardnegative_train_contriever.jsonl" \
        --output_dir "dpr_trained/${dataset}/${case}_seed${seed}" \
        --batch_size 32 \
        --save_steps $save_steps \
        --lr 2e-5 \
        --epoch $epochs \
        --seed $seed
fi

# only evaluate the last checkpoint if eval_last_only flag is on
if [ "$eval_last_only" = true ]; then
    subdir=$(ls -d dpr_trained/${dataset}/${case}_seed${seed}/*/ | grep checkpoint_ | sort -V | tail -n 1)
    if [ -d "$subdir" ]; then
        if [ ! -f "${subdir}/ndcg.json" ]; then
            echo "Evaluating dpr for ${dataset}... ${case}-${subdir}"
            python dpr_finetuning/test_dpr.py \
                --model_dir "$subdir" \
                --beir_dataset_path "datasets/${dataset}" \
                --batch_size 128
        fi
    fi
else
    for subdir in $(echo "dpr_trained/${dataset}/${case}_seed${seed}"/*/ | tr ' ' '\n' | sort -V | tr '\n' ' '); do
        if [ -d "$subdir" ]; then
            if [ ! -f "${subdir}/ndcg.json" ]; then
                echo "Evaluating dpr for ${dataset}... ${case}-${subdir}"
            python dpr_finetuning/test_dpr.py \
                --model_dir "$subdir" \
                --beir_dataset_path "datasets/${dataset}" \
                --batch_size 128
            fi
        fi
    done
fi