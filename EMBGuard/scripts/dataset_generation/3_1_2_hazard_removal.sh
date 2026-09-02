#!/bin/bash
# Step 3-1-2: Hazard Removal Pipeline - New Task-based Version
# Usage: Edit configuration below and run: ./scripts/3_1_2_hazard_removal_new.sh

set -e  # Exit on any error

# ============================================
# Configuration - EDIT THESE
# ============================================

# Iteration name (results folder name)
ITERATE_NAME="train"

# Input graphs file (relative to iterate directory or absolute path)
# If not set, will auto-detect: graphs_scene_augmented.json, graphs_normalized.json, or graphs.json
INPUT_GRAPHS_FILE="graphs_scene_augmented.json"  # e.g., "graphs_scene_augmented.json" or "graphs_normalized.json"

# Model settings
HAZARD_REMOVAL_MODEL="gpt-5.2"
HAZARD_REMOVAL_PROVIDER="openai"

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

# ============================================
# Pipeline Execution
# ============================================

echo "Hazard Removal Pipeline (New): ${ITERATE_NAME}"
mkdir -p "${ITERATE_DIR}"

# Run hazard_removal (gathering is handled automatically)
echo "Running hazard removal..."
INPUT_FILE_ARG=""
if [ -n "${INPUT_GRAPHS_FILE}" ]; then
  INPUT_FILE_ARG="--data_path ${INPUT_GRAPHS_FILE}"
fi

python ${PROJECT_ROOT}/src/run_dataset_generation.py \
  --task hazard_removal \
  --model_name "${HAZARD_REMOVAL_MODEL}" \
  --provider "${HAZARD_REMOVAL_PROVIDER}" \
  --save_dir "${RESULTS_DIR}" \
  --iterate_name "${ITERATE_NAME}/raw" \
  --num_workers ${NUM_WORKERS} \
  ${INPUT_FILE_ARG}

# Summary
echo ""
echo "✓ Hazard Removal completed: ${ITERATE_DIR}"
if [ -n "${INPUT_GRAPHS_FILE}" ]; then
  # Extract base name and generate output filename
  INPUT_BASENAME=$(basename "${INPUT_GRAPHS_FILE}" .json)
  OUTPUT_FILE="${INPUT_BASENAME}_hazard_removed.json"
  echo "  - ${OUTPUT_FILE}"
else
  echo "  - graphs_hazard_removed.json"
fi

