#!/bin/bash
# Step 5-1: Text to Image Pipeline (with batch) - New Task-based Version
# Usage: Edit configuration below and run: ./scripts/5_1_text_to_image_w_batch_new.sh

set -e  # Exit on any error

# ============================================
# Configuration - EDIT THESE
# ============================================

# Iteration name (results folder name)
ITERATE_NAME="train"

# Model settings
IMAGE_MODEL="gemini-3-pro-image-preview"
IMAGE_PROVIDER="gemini"  # For batch API, use gemini

# Split by merge_source and create one batch job per source
SPLIT_BY_SOURCE="true"
SPLIT_DIR_NAME="texts_by_merge_source"
SLEEP_BETWEEN_BATCH_SEC=2

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
INPUT_PATH="${ITERATE_DIR}/texts_generated.json"


SPLIT_OUTPUT_DIR="${ITERATE_DIR}/${SPLIT_DIR_NAME}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%s)}"
BATCH_BASE_DIR="${ITERATE_DIR}/${BATCH_BASE_DIR_NAME}"
BATCH_RUN_DIR="${BATCH_BASE_DIR}/${RUN_TIMESTAMP}"

# ============================================
# Pipeline Execution
# ============================================

echo "Text to Image Pipeline (New): ${ITERATE_NAME}"
mkdir -p "${ITERATE_DIR}"

# Split input by merge_source (optional)
RUN_FILES=("${INPUT_PATH}")
if [ "${SPLIT_BY_SOURCE}" = "true" ]; then
  echo "Splitting input by merge_source..."
  python ${PROJECT_ROOT}/src/dataset_generation/utils/split_by_merge_source.py \
    --input "${INPUT_PATH}" \
    --output-dir "${SPLIT_OUTPUT_DIR}" \
    --output-template "texts_{merge_source}.json"

  shopt -s nullglob
  RUN_FILES=("${SPLIT_OUTPUT_DIR}"/texts_*.json)
  if [ ${#RUN_FILES[@]} -eq 0 ]; then
    echo "Error: No split files found under: ${SPLIT_OUTPUT_DIR}"
    exit 1
  fi
fi

# Create batch job(s) for image generation (one job per file)
echo "Creating batch job(s) for image generation..."
mkdir -p "${BATCH_RUN_DIR}"
echo "Batch output dir: ${BATCH_RUN_DIR}"
for file in "${RUN_FILES[@]}"; do
  echo ""
  echo "Input file: ${file}"
  echo "Number of scenarios: $(python -c "import json; d=json.load(open('${file}')); print(len(d.get('scenarios', d)))")"

  # Extract merge_source from filename for logging
  CURRENT_MERGE_SOURCE=$(python -c "
import os
filename = os.path.basename('${file}')
if filename.startswith('texts_') and filename.endswith('.json'):
    merge_source = filename[6:-5]  # Remove 'texts_' and '.json'
    print(merge_source)
else:
    print('unknown')
")
  echo "Processing merge_source: ${CURRENT_MERGE_SOURCE}"

  python ${PROJECT_ROOT}/src/run_dataset_generation.py \
    --task text_to_image \
    --model_name "${IMAGE_MODEL}" \
    --provider "${IMAGE_PROVIDER}" \
    --data_path "${file}" \
    --save_dir "${RESULTS_DIR}" \
    --iterate_name "${ITERATE_NAME}/${BATCH_BASE_DIR_NAME}/${RUN_TIMESTAMP}" \
    --batch True

  sleep "${SLEEP_BETWEEN_BATCH_SEC}"
done

# Step 2: gather batch scenario information
echo "Gathering batch scenario information..."
python ${PROJECT_ROOT}/src/dataset_generation/batch_job/gather_batch_scenarios.py \
  --raw-dir "${BATCH_RUN_DIR}" \
  --output "${ITERATE_DIR}/batch_scenarios.json"

# Summary
echo ""
echo "✓ Batch job creation completed: ${ITERATE_DIR}"
echo "  - Batch job created and submitted to Google AI"
echo "  - Batch scenarios info: batch_scenarios.json"
echo "  - Raw files in: raw/"
echo ""
echo "Next steps:"
echo "  1. Wait for batch job to complete (check status periodically)"
echo "  2. Run 5_2_check_batch_new.sh to download results"
echo "  3. Run 5_3_download_image_new.sh to extract images"

