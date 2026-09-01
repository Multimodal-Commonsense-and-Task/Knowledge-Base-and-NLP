#!/bin/bash
SCRIPT_DIR=$(dirname "$0")
set -e
dataset=$1
entropy=$2
contamination=${3:-1.5}
# range of iterations
for i in {0..9}
do
    $SCRIPT_DIR/run_pipeline-dpr-iter.sh $dataset $entropy $i
    # Capture the exit code
    exit_code=$?

    if [ $exit_code -eq 1 ]; then
        echo "Early stopped at iteration $((i-1))"
        exit 0
    fi
done