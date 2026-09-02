#!/bin/bash
# Step 3-1-1: Scene Augmentation Pipeline - New Task-based Version
# Usage: Edit configuration below and run: ./scripts/3_1_1_scene_augmentation_new.sh

set -e  # Exit on any error

# ============================================
# Configuration - EDIT THESE
# ============================================

# Iteration name (results folder name)
ITERATE_NAME="train"

# Model settings
SCENE_AUG_MODEL="gpt-5.2"
SCENE_AUG_PROVIDER="openai"

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

echo "Scene Augmentation Pipeline (New): ${ITERATE_NAME}"
mkdir -p "${ITERATE_DIR}"

# Run scene_augmentation (gathering is handled automatically)
echo "Running scene augmentation..."
python ${PROJECT_ROOT}/src/run_dataset_generation.py \
  --task scene_augmentation \
  --model_name "${SCENE_AUG_MODEL}" \
  --provider "${SCENE_AUG_PROVIDER}" \
  --save_dir "${RESULTS_DIR}" \
  --iterate_name "${ITERATE_NAME}/raw" \
  --num_workers ${NUM_WORKERS}

# Summary
echo ""
echo "✓ Scene Augmentation completed: ${ITERATE_DIR}"
echo "  - graphs_scene_augmented.json"

