#!/bin/bash
# Step 2: Make Graphs Pipeline - New Task-based Version
# Usage: Edit configuration below and run: ./scripts/2_make_graphs_new.sh

set -e  # Exit on any error

# ============================================
# Configuration - EDIT THESE
# ============================================

# Iteration name (results folder name)
ITERATE_NAME="train"

# Input scenarios file (relative to iterate directory or absolute path)
# If not set, will auto-detect: scenarios_generated.json or scenarios.json
SCENARIOS_FILE="scenarios_generated.json"  # e.g., "scenarios_generated.json" or "scenarios.json"

# Model settings per step
# Step 1: scenario → graph
GRAPH_MODEL="gpt-5.2"
GRAPH_PROVIDER="openai"

# Step 2: scene normalization
NORM_MODEL="gpt-5.2"
NORM_PROVIDER="openai"

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

echo "Make Graphs (New): ${ITERATE_NAME}"
mkdir -p "${ITERATE_DIR}"

# Step 1: scenario → graph (gathering is handled automatically)
echo "[1/2] Extracting graphs..."
SCENARIOS_FILE_ARG=""
if [ -n "${SCENARIOS_FILE}" ]; then
  SCENARIOS_FILE_ARG="--scenarios_file ${SCENARIOS_FILE}"
fi

python ${PROJECT_ROOT}/src/run_dataset_generation.py \
  --task scenario_to_graph \
  --model_name "${GRAPH_MODEL}" \
  --provider "${GRAPH_PROVIDER}" \
  --save_dir "${RESULTS_DIR}" \
  --iterate_name "${ITERATE_NAME}/raw" \
  --num_workers ${NUM_WORKERS} \
  ${SCENARIOS_FILE_ARG}

# Step 2: scene normalization (gathering is handled automatically)
echo "[2/2] Normalizing scenes..."
python ${PROJECT_ROOT}/src/run_dataset_generation.py \
  --task scene_normalization \
  --model_name "${NORM_MODEL}" \
  --provider "${NORM_PROVIDER}" \
  --save_dir "${RESULTS_DIR}" \
  --iterate_name "${ITERATE_NAME}/raw" \
  --num_workers ${NUM_WORKERS}

# Summary
echo ""
echo "✓ Make Graphs completed: ${ITERATE_DIR}"
echo "  - graphs.json"
echo "  - graphs_normalized.json"

