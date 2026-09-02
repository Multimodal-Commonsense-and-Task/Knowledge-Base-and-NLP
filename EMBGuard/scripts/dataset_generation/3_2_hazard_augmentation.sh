#!/bin/bash
# Step 3-2: Hazard Augmentation Pipeline - New Task-based Version
# Usage: Edit configuration below and run: ./scripts/3_2_hazard_augmentation_new.sh

set -e  # Exit on any error

# ============================================
# Configuration - EDIT THESE
# ============================================

# Iteration name (results folder name)
ITERATE_NAME="train"

# Input scenarios file (relative to iterate directory or absolute path)
# If not set, will auto-detect: graphs_normalized.json or graphs.json
INPUT_SCENARIOS_FILE="graphs_normalized.json"  # e.g., "graphs_normalized.json" or "graphs_scene_augmented.json"

# Model settings
HAZARD_AUG_MODEL="gpt-4.1-mini"
HAZARD_AUG_PROVIDER="openai"

# Sampling settings
RANDOM_SEED=42
PAIRS_PER_MECHANISM=5  # Number of pairs to select per mechanism combination

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

echo "Hazard Augmentation Pipeline: ${ITERATE_NAME}"
mkdir -p "${ITERATE_DIR}"

# Run hazard_augmentation (pairs creation, inference, and splitting are all handled automatically)
echo "Running hazard augmentation (pairs creation, inference, and splitting)..."
INPUT_FILE_ARG=""
if [ -n "${INPUT_SCENARIOS_FILE}" ]; then
  INPUT_FILE_ARG="--input_scenarios_file ${INPUT_SCENARIOS_FILE}"
fi

python ${PROJECT_ROOT}/src/run_dataset_generation.py \
  --task hazard_augmentation \
  --model_name "${HAZARD_AUG_MODEL}" \
  --provider "${HAZARD_AUG_PROVIDER}" \
  --save_dir "${RESULTS_DIR}" \
  --iterate_name "${ITERATE_NAME}/raw" \
  --num_workers ${NUM_WORKERS} \
  --random_seed ${RANDOM_SEED} \
  --pairs_per_mechanism ${PAIRS_PER_MECHANISM} \
  ${INPUT_FILE_ARG}

# Summary
echo ""
echo "✓ Hazard Augmentation completed: ${ITERATE_DIR}"
echo "  - raw/hazard_pairs_by_room.json (mechanism coverage pairs)"
echo "  - raw/hazard_augmentation.json (inference results)"

# Determine output filenames based on input
if [ -n "${INPUT_SCENARIOS_FILE}" ]; then
  INPUT_BASENAME=$(basename "${INPUT_SCENARIOS_FILE}" .json)
  OUTPUT_FILE="${INPUT_BASENAME}_hazard_augmented.json"
  SPLIT_FILE="${INPUT_BASENAME}_hazard_augmented_split.json"
else
  OUTPUT_FILE="graphs_hazard_augmented.json"
  SPLIT_FILE="graphs_hazard_augmented_split.json"
fi

echo "  - ${OUTPUT_FILE} (dual structure scenarios)"
echo "  - ${SPLIT_FILE} (single scenarios)"
echo ""
echo "Configuration used:"
if [ -n "${INPUT_SCENARIOS_FILE}" ]; then
  echo "  - Input: ${INPUT_SCENARIOS_FILE}"
else
  echo "  - Input: graphs_normalized.json (auto-detected)"
fi
echo "  - Random seed: ${RANDOM_SEED}"
echo "  - Pairs per mechanism: ${PAIRS_PER_MECHANISM}"

