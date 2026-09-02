#!/bin/bash
#SBATCH --job-name=colbert_training
#SBATCH --output=slurm_output_train-colbert.txt
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
set -e



# Directory where you keep the beir-ColBERT repo
cd /your/path/beir-ColBERT
# source ~/.bashrc
# # Set your conda virtual environment name
# conda activate conda_env_name
source venv/bin/activate

dataset=$1

python -m colbert.data_prep \
    --dataset ${dataset} \
    --split "dev" \
    --data_dir "datasets/${dataset}" \
    --collection "datasets/${dataset}-collection-dev.tsv" \
    --queries "datasets/${dataset}-queries-dev.tsv"