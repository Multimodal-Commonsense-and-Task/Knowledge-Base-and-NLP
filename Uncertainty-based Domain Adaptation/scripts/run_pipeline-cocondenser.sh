#!/bin/bash
set -e
SCRIPT_DIR=$(dirname "$0")
dataset=$1
entropy=$2
lr=${3:-5e-6}
contamination=${4:-1.5}
# range of iterations
for i in {0..9}
do
    $SCRIPT_DIR/scripts/run_pipeline-cocondenser-iter.sh $dataset $entropy $i $lr $contamination
    # Capture the exit code
    exit_code=$?

    if [ $exit_code -eq 1 ]; then
        echo "Early stopped at iteration $((i-1))"
        exit 0
    fi
done