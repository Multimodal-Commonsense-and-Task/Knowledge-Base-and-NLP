#!/bin/bash
SCRIPT_DIR=$(dirname "$0")

dataset=$1
case=$2
full=$3

python $SCRIPT_DIR/data_preparation/query_generation/duqgen.py --range 0 100000 --fill_holes True \
    --job_name '${dataset} DUQ Gen' --semaphore 8 --case _$case --datasets $dataset \