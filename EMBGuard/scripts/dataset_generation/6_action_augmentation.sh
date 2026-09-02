#!/bin/bash
# Action Augmentation Pipeline - New Task-based Version
# - Runs `action_augmentation` inference (safe action rewrite)
# - Gathers augmented scenarios by replacing `action` while keeping the same graph/hazard
#
# Usage: Edit configuration below and run: ./scripts/6_action_augmentation_new.sh

set -e  # Exit on any error

# ============================================
# Configuration - EDIT THESE
# ============================================

# Iteration name (results folder name)
ITERATE_NAME="train"

# Input file (optional, will auto-detect texts_with_images_*.json if not provided)
# Must include: id, hazard, action, graph, image_path
# Leave empty to auto-detect, or provide relative path from iterate_dir or absolute path
INPUT_FILE="texts_with_images_scene_augmented.json"

# Model settings
ACTION_AUG_MODEL="gpt-5.2"
ACTION_AUG_PROVIDER="openai"

# Whether to include original scenarios in output
# - true: output contains originals + action-augmented
# - false: output contains action-augmented only
INCLUDE_ORIGINAL=false

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

echo "Action Augmentation Pipeline (New): ${ITERATE_NAME}"
mkdir -p "${ITERATE_DIR}"

# Prepare data_path argument
DATA_PATH_ARG=""
if [ -n "${INPUT_FILE}" ]; then
    # Resolve input path (absolute or relative to iterate_dir)
    if [[ "${INPUT_FILE}" == /* ]]; then
        INPUT_PATH="${INPUT_FILE}"
    else
        INPUT_PATH="${ITERATE_DIR}/${INPUT_FILE}"
    fi
    
    if [ ! -f "${INPUT_PATH}" ]; then
        echo "Error: Input file not found: ${INPUT_PATH}"
        exit 1
    fi
    DATA_PATH_ARG="--data_path ${INPUT_PATH}"
    echo "Using input file: ${INPUT_PATH}"
fi

# Run action_augmentation inference (gathering is handled automatically)
echo "Running action augmentation..."
python ${PROJECT_ROOT}/src/run_dataset_generation.py \
  --task action_augmentation \
  --model_name "${ACTION_AUG_MODEL}" \
  --provider "${ACTION_AUG_PROVIDER}" \
  --save_dir "${RESULTS_DIR}" \
  --iterate_name "${ITERATE_NAME}/raw" \
  --num_workers ${NUM_WORKERS} \
  ${DATA_PATH_ARG} \
  --include_original ${INCLUDE_ORIGINAL}

echo ""
echo "✓ Action Augmentation completed: ${ITERATE_DIR}"

