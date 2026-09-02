#!/bin/bash
SCRIPT_DIR=$(dirname "$0")

dataset=$1
case=$2
summary=$3

$SCRIPT_DIR/qgen_merge.sh $dataset $case $summary
case="merged_${summary}_${case}"

$SCRIPT_DIR/hard_negative.sh $dataset $case
$SCRIPT_DIR/run_epochs.sh $dataset $case true 1
$SCRIPT_DIR/print_epoch_results.sh $dataset $case true