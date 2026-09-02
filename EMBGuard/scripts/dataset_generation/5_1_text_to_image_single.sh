#!/bin/bash
# Step 5-1: Text to Image Pipeline (single file) - Batch Version
# Usage: Edit configuration below and run: ./scripts/5_1_text_to_image_single.sh
# This script runs batch job for a single input file (no splitting)

set -e  # Exit on any error

# ============================================
# Configuration - EDIT THESE
# ============================================

# Iteration name (results folder name)
ITERATE_NAME="train"

# Input file (relative to iterate directory or absolute path)
# Example: "texts_by_merge_source/texts_original.json"
INPUT_FILE="texts_by_merge_source/texts_scene_augmented.json"

# Model settings
IMAGE_MODEL="gemini-3-pro-image-preview"
IMAGE_PROVIDER="gemini"  # For batch API, use gemini

# Where to write batch artifacts (relative to iterate dir)
BATCH_BASE_DIR_NAME="raw/batch"

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

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%s)}"
BATCH_BASE_DIR="${ITERATE_DIR}/${BATCH_BASE_DIR_NAME}"
BATCH_RUN_DIR="${BATCH_BASE_DIR}/${RUN_TIMESTAMP}"

# ============================================
# Validation
# ============================================

echo "Text to Image Pipeline (Single File): ${ITERATE_NAME}"
echo "Input file: ${INPUT_PATH}"
mkdir -p "${ITERATE_DIR}"

# Check if input file exists
if [ ! -f "${INPUT_PATH}" ]; then
    echo "Error: Input file not found: ${INPUT_PATH}"
    exit 1
fi

# ============================================
# Pipeline Execution
# ============================================

# Extract merge_source from filename for logging
CURRENT_MERGE_SOURCE=$(python -c "
import os
filename = os.path.basename('${INPUT_PATH}')
if filename.startswith('texts_') and filename.endswith('.json'):
    merge_source = filename[6:-5]  # Remove 'texts_' and '.json'
    print(merge_source)
else:
    print('unknown')
")

echo "Processing merge_source: ${CURRENT_MERGE_SOURCE}"
echo "Number of scenarios: $(python -c "import json; d=json.load(open('${INPUT_PATH}')); print(len(d.get('scenarios', d)))")"
echo ""

# Create batch job for the single file
echo "Creating batch job for image generation..."
mkdir -p "${BATCH_RUN_DIR}"
echo "Batch output dir: ${BATCH_RUN_DIR}"

python ${PROJECT_ROOT}/src/run_dataset_generation.py \
  --task text_to_image \
  --model_name "${IMAGE_MODEL}" \
  --provider "${IMAGE_PROVIDER}" \
  --data_path "${INPUT_PATH}" \
  --save_dir "${RESULTS_DIR}" \
  --iterate_name "${ITERATE_NAME}/${BATCH_BASE_DIR_NAME}/${RUN_TIMESTAMP}" \
  --batch True

# Step 2: gather batch scenario information (optional, only if batch_scenarios.json doesn't exist or you want to update)
echo ""
echo "Gathering batch scenario information..."
python ${PROJECT_ROOT}/src/dataset_generation/batch_job/gather_batch_scenarios.py \
  --raw-dir "${BATCH_RUN_DIR}" \
  --output "${ITERATE_DIR}/batch_scenarios.json"

# Summary
echo ""
echo "✓ Batch job creation completed: ${ITERATE_DIR}"
echo "  - Batch job created and submitted to Google AI"
echo "  - Input file: ${INPUT_FILE}"
echo "  - Merge source: ${CURRENT_MERGE_SOURCE}"
echo "  - Batch scenarios info: batch_scenarios.json"
echo "  - Raw files in: ${BATCH_RUN_DIR}"
echo ""
echo "Next steps:"
echo "  1. Wait for batch job to complete (check status periodically)"
echo "  2. Run 5_2_check_batch.sh to download results"
echo "  3. Run 5_3_download_image.sh to extract images"
