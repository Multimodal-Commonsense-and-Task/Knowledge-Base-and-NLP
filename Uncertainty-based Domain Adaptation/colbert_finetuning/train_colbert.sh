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
# source venv/bin/activate




dataset=$1
case=$2
# Input data file to train
train_filepath="/your/path/datasets/${dataset}_${case}_hardnegative_train_colbert.tsv"
# Directory to save the trained colbert model
trained_root=/your/path/colbert_trained/${dataset}/${case}
# Path to the MS-MARCO pre-trained ColBERT checkpoint
checkpoint_path=/your/path/beir-ColBERT/trained_models/colbert/colbert-300000.dnn




python -m colbert.train \
            --amp --doc_maxlen 300 --mask-punctuation --bsize 32 --accum 1 \
            --triples ${train_filepath} \
            --root ${trained_root} --experiment MSMARCO-psg --similarity l2 \
            --lr 3e-6 \
            --checkpoint $checkpoint_path