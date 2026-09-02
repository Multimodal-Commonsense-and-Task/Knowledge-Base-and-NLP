#!/bin/bash
# Step 4: Graph to Text Pipeline - New Task-based Version
# Usage: Edit configuration below and run: ./scripts/4_graph_to_text_new.sh

set -e  # Exit on any error

# ============================================
# Configuration - EDIT THESE
# ============================================

# Iteration name (results folder name)
ITERATE_NAME="train"

# Input file (relative to iterate directory or absolute path)
INPUT_FILE="graphs_final_for_image_generation.json"

# Model settings
TEXT_MODEL="gpt-5.2"
TEXT_PROVIDER="openai"

# Number of worker processes
NUM_WORKERS=24

# ============================================
# Auto-configured paths
# ============================================

# Get project root (assuming script is in scripts/dataset_generation/ folder)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

RESULTS_DIR="${PROJECT_ROOT}/dataset_generation_output"
ITERATE_DIR="${RESULTS_DIR}/${ITERATE_NAME}"

# Resolve input file path
if [[ "${INPUT_FILE}" == /* ]]; then
    # Absolute path
    INPUT_PATH="${INPUT_FILE}"
else
    # Relative to iterate directory
    INPUT_PATH="${ITERATE_DIR}/${INPUT_FILE}"
fi

# ============================================
# Pipeline Execution
# ============================================

echo "Graph to Text Pipeline (New): ${ITERATE_NAME}"
echo "Input file: ${INPUT_PATH}"
mkdir -p "${ITERATE_DIR}"

# Check if input file exists
if [ ! -f "${INPUT_PATH}" ]; then
    echo "Error: Input file not found: ${INPUT_PATH}"
    exit 1
fi

# Convert graphs to text descriptions (gathering is handled automatically)
echo "Converting graphs to text descriptions..."
python ${PROJECT_ROOT}/src/run_dataset_generation.py \
  --task graph_to_text \
  --model_name "${TEXT_MODEL}" \
  --provider "${TEXT_PROVIDER}" \
  --save_dir "${RESULTS_DIR}" \
  --iterate_name "${ITERATE_NAME}/raw" \
  --data_path "${INPUT_PATH}" \
  --num_workers ${NUM_WORKERS}

# Summary
echo ""
echo "✓ Graph to Text completed: ${ITERATE_DIR}"
echo "  - texts_generated.json"

