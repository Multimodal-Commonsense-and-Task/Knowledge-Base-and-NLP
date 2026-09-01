#!/bin/bash
set -e
SCRIPT_DIR=$(dirname "$0")

dataset=$1
entropy=$2
lr=${3:-5e-6}
contamination=${4:-1.5}

for i in {0..9}
do
    $SCRIPT_DIR/run_pipeline-qwen3-iter.sh "$dataset" "$entropy" "$i" "$lr" "$contamination"
    exit_code=$?

    if [ $exit_code -eq 1 ]; then
        echo "Early stopped at iteration $((i-1))"
        exit 0
    fi
done
