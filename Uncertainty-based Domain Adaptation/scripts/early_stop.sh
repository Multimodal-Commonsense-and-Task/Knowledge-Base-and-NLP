#!/bin/bash
SCRIPT_DIR=$(dirname "$0")
set -e

dataset=$1
model=$2
case=$3
alpha=${4:-0.4}

# Capture the output from the Python script
output=$(python $SCRIPT_DIR/scripts/early_stop.py \
    --dataset $dataset \
    --model $model \
    --case_pattern $case \
    --alpha $alpha)

# Extract the true/false value from the output
# Look for the line containing "Early stop detected" and extract True/False
early_stop=$(echo "$output" | grep "Early stop detected" | awk '{print $NF}')

# Print the full output for debugging
echo "$output"
echo ""
echo "================================"
echo "Early stop result: $early_stop"

if [ "$early_stop" = "True" ]; then
    echo "Early stopping triggered - local minima detected!"
    exit 0
else
    echo "No early stopping - continue training"
    exit 1
fi