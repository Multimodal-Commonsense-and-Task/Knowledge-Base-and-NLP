#!/bin/bash
# Step 4: Graph to Text - Gather Results Only
# Usage: Edit configuration below and run: ./scripts/4_graph_to_text_gather.sh
# This script only runs the gather step, assuming inference is already complete.

set -e  # Exit on any error

# ============================================
# Configuration - EDIT THESE
# ============================================

# Iteration name (results folder name)
ITERATE_NAME="train"

# Input file (relative to iterate directory or absolute path)
# This should be the same file used for inference
INPUT_FILE="graphs_final_for_image_generation.json"

# Results file (inference output, relative to iterate directory or absolute path)
# Default: raw/graph_to_text.json
RESULTS_FILE="raw/graph_to_text.json"

# Output file (final gathered results, relative to iterate directory or absolute path)
# Default: texts_generated.json
OUTPUT_FILE="texts_generated.json"

# ============================================
# Auto-configured paths
# ============================================

# Get project root (assuming script is in scripts/dataset_generation/ folder)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

RESULTS_DIR="${PROJECT_ROOT}/dataset_generation_output"
ITERATE_DIR="${RESULTS_DIR}/${ITERATE_NAME}"

# Resolve input file path (scenarios file)
if [[ "${INPUT_FILE}" == /* ]]; then
    # Absolute path
    SCENARIOS_FILE="${INPUT_FILE}"
else
    # Relative to iterate directory
    SCENARIOS_FILE="${ITERATE_DIR}/${INPUT_FILE}"
fi

# Resolve results file path (inference output)
if [[ "${RESULTS_FILE}" == /* ]]; then
    # Absolute path
    RESULTS_PATH="${RESULTS_FILE}"
else
    # Relative to iterate directory
    RESULTS_PATH="${ITERATE_DIR}/${RESULTS_FILE}"
fi

# Resolve output file path
if [[ "${OUTPUT_FILE}" == /* ]]; then
    # Absolute path
    OUTPUT_PATH="${OUTPUT_FILE}"
else
    # Relative to iterate directory
    OUTPUT_PATH="${ITERATE_DIR}/${OUTPUT_FILE}"
fi

# ============================================
# Validation
# ============================================

echo "Graph to Text - Gather Results Only: ${ITERATE_NAME}"
echo "Scenarios file: ${SCENARIOS_FILE}"
echo "Results file: ${RESULTS_PATH}"
echo "Output file: ${OUTPUT_PATH}"
echo ""

# Check if files exist
if [ ! -f "${SCENARIOS_FILE}" ]; then
    echo "Error: Scenarios file not found: ${SCENARIOS_FILE}"
    exit 1
fi

if [ ! -f "${RESULTS_PATH}" ]; then
    echo "Error: Results file not found: ${RESULTS_PATH}"
    exit 1
fi

# ============================================
# Gather Results
# ============================================

echo "Running gather_results..."
python -c "
import sys
import os
sys.path.insert(0, '${PROJECT_ROOT}')

from src.dataset_generation.task import get_task
import argparse

# Create minimal args object
args = argparse.Namespace()
args.task = 'graph_to_text'
args.prompt_path = None  # Not needed for gather
args.shots_path = None  # Not needed for gather
args.data_path = '${SCENARIOS_FILE}'
args.save_dir = '${RESULTS_DIR}'
args.iterate_name = '${ITERATE_NAME}/raw'

# Get task instance
task = get_task('graph_to_text', args)

# Run gather_results
result = task.gather_results(
    results_file='${RESULTS_PATH}',
    scenarios_file='${SCENARIOS_FILE}',
    output_file='${OUTPUT_PATH}'
)

print(f'\\n✓ Gather completed successfully')
print(f'  Total scenarios: {result[\"metadata\"][\"total_scenarios\"]}')
print(f'  Scenarios with texts: {result[\"metadata\"][\"scenarios_with_texts\"]}')
print(f'  Scenarios without texts: {result[\"metadata\"][\"scenarios_without_texts\"]}')
print(f'  Success rate: {result[\"metadata\"][\"success_rate\"]:.2%}')
"

# Summary
echo ""
echo "✓ Gather Results completed: ${ITERATE_DIR}"
echo "  - Output: ${OUTPUT_FILE}"
