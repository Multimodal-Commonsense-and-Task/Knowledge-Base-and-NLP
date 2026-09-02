#!/bin/bash
# Step 1: Base Pipeline (taxonomy → scenario) - New Task-based Version
# Usage: Edit configuration below and run: ./scripts/1_make_scenarios_new.sh

set -e  # Exit on any error

# ============================================
# Configuration - EDIT THESE
# ============================================

# Iteration name (results folder name)
ITERATE_NAME="train"

# Number of scenarios to generate (0 = use shots only, no generation)
NUM_SCENARIOS=100

# Model settings per step
# taxonomy → scenario
TAXONOMY_MODEL="gpt-5.2"
TAXONOMY_PROVIDER="openai"  # openai, openrouter, vllm, claude, gemini

# Number of worker processes
NUM_WORKERS=24

# Generation configs
TAXONOMY_TEMPERATURE=0.7
TAXONOMY_MAX_TOKENS=32768

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

echo "Base Pipeline (New): ${ITERATE_NAME}"
mkdir -p "${ITERATE_DIR}"

# Generate scenarios from taxonomy (gathering is handled automatically by the task)
echo "Generating scenarios from taxonomy per risk/mechanism..."
RAW_DIR="${RESULTS_DIR}/${ITERATE_NAME}/raw"
mkdir -p "${RAW_DIR}"

python ${PROJECT_ROOT}/src/run_dataset_generation.py \
  --task taxonomy_to_scenario \
  --model_name "${TAXONOMY_MODEL}" \
  --provider "${TAXONOMY_PROVIDER}" \
  --save_dir "${RESULTS_DIR}" \
  --iterate_name "${ITERATE_NAME}/raw" \
  --num ${NUM_SCENARIOS} \
  --num_workers ${NUM_WORKERS} \
  --temperature ${TAXONOMY_TEMPERATURE} \
  --max_tokens ${TAXONOMY_MAX_TOKENS}

# Summary
echo ""
echo "✓ Make Scenarios completed: ${ITERATE_DIR}"
echo "  - scenarios.json"

