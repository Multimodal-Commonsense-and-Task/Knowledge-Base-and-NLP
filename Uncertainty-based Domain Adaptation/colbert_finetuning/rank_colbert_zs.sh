#!/bin/bash
#SBATCH --job-name=colbert_testing
#SBATCH --output=slurm_output_test-colbert.txt
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB

set -e




# Directory where you keep the beir-ColBERT repo
cd /your/path/beir-ColBERT
# source ~/.bashrc
# Set your conda virtual environment
# conda activate conda_env_name
source venv/bin/activate


# Dataset name
dataset=$1
case=$2
NUM_PARTITIONS=32768
top_dir=/your/path/beir-ColBERT
# Prepared target collection & test queries in the tsv format (please refer to beir-ColBERT for the data formatting: https://github.com/thakur-nandan/beir-ColBERT)
COLLECTION=$top_dir/datasets/${dataset}-collection.tsv
QUERIES=$top_dir/datasets/${dataset}-queries-${case}.tsv
INDEX_NAME=${dataset}-colbert-${case}
# Path to the trained colbert model (only needs the directory having *.dnn file)
CHECKPOINT=$top_dir/trained_models/colbert/colbert-300000.dnn
## if we want to automatically detect the *.dnn file inside the directory, then use the below line
# CHECKPOINT=$(find "${top_dir}/trained_models/${dataset}/${case}/MSMARCO-psg/train.py/" -name "colbert-*.dnn" -type f)
echo "Loading the checkpoint file >>> $CHECKPOINT"

ROOT_DIR=/your/path/colbert_trained/${dataset}/zs/${case}
OUTPUT_DIR=${ROOT_DIR}/output
INDEX_ROOT=${ROOT_DIR}/index
RANKING_DIR=${ROOT_DIR}/ranking

#####################################################################################################################################
#                                                                 (1) Indexing
#####################################################################################################################################

# only perform when output folder does not exists
python -m colbert.index \
  --root $OUTPUT_DIR \
  --doc_maxlen 300 \
  --mask-punctuation \
  --bsize 128 \
  --amp \
  --checkpoint $CHECKPOINT \
  --index_root $INDEX_ROOT \
  --index_name $INDEX_NAME \
  --collection $COLLECTION \
  --experiment ${dataset}

#####################################################################################################################################
#                                                                 (2) Faiss Indexing
#####################################################################################################################################

python -m colbert.index_faiss \
  --index_root $INDEX_ROOT \
  --index_name $INDEX_NAME \
  --partitions $NUM_PARTITIONS \
  --sample 0.3 \
  --root $OUTPUT_DIR \
  --experiment ${dataset}

# #####################################################################################################################################
# #                                                                 (3) ANN Search
# #####################################################################################################################################
if [ ! -d "$RANKING_DIR" ]; then
  python -m colbert.retrieve \
    --amp \
    --doc_maxlen 300 \
    --mask-punctuation \
    --bsize 256 \
    --queries $QUERIES \
    --nprobe 32 \
    --partitions $NUM_PARTITIONS \
    --faiss_depth 100 \
    --depth 10 \
    --index_root $INDEX_ROOT \
    --index_name $INDEX_NAME \
    --checkpoint $CHECKPOINT \
    --root $OUTPUT_DIR \
    --experiment ${dataset} \
    --ranking_dir $RANKING_DIR
fi

echo "===================================================================================="
echo "              Completed testing with dataset: ${dataset}"
echo "===================================================================================="
