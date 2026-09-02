#!/bin/bash
# Step 5-3: Download Images from Batch Job Results - New Version
# Usage: Edit configuration below and run: ./scripts/5_3_download_image_new.sh

set -e  # Exit on any error

# ============================================
# Configuration - EDIT THESE
# ============================================

# Iteration name (results folder name)
ITERATE_NAME="train"

# Required: set this to the run timestamp directory under raw/batch/
# This should match the timestamp used in 5_1_text_to_image_w_batch_new.sh
# Leave empty to auto-detect the latest timestamp
BATCH_RUN_TIMESTAMP="1768364195"

# ============================================
# Auto-configured paths
# ============================================

# Get project root (assuming script is in scripts/dataset_generation/ folder)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

RESULTS_DIR="${PROJECT_ROOT}/dataset_generation_output"
ITERATE_DIR="${RESULTS_DIR}/${ITERATE_NAME}"
BATCH_BASE_DIR="${ITERATE_DIR}/raw/batch"
OUTPUT_DIR="${ITERATE_DIR}/images/downloaded/augmented"

# Auto-detect latest timestamp if not provided
if [ -z "${BATCH_RUN_TIMESTAMP}" ]; then
    # Find the latest timestamp directory
    LATEST_TIMESTAMP=$(ls -td "${BATCH_BASE_DIR}"/[0-9]* 2>/dev/null | head -1 | xargs basename)
    if [ -z "${LATEST_TIMESTAMP}" ]; then
        echo "Error: No batch timestamp directory found in ${BATCH_BASE_DIR}"
        echo "Please set BATCH_RUN_TIMESTAMP manually"
        exit 1
    fi
    BATCH_RUN_TIMESTAMP="${LATEST_TIMESTAMP}"
    echo "Auto-detected latest timestamp: ${BATCH_RUN_TIMESTAMP}"
fi

RUN_DIR="${BATCH_BASE_DIR}/${BATCH_RUN_TIMESTAMP}"

# ============================================
# Pipeline Execution
# ============================================

echo "Downloading Images from Batch Job Results: ${ITERATE_NAME}"
echo "Using timestamp: ${BATCH_RUN_TIMESTAMP}"
echo "Output directory: ${OUTPUT_DIR}"

# Find all batch result files
BATCH_RESULT_FILES=("${RUN_DIR}"/batch_result_*.jsonl)
if [ ${#BATCH_RESULT_FILES[@]} -eq 0 ] || [ ! -f "${BATCH_RESULT_FILES[0]}" ]; then
    echo "Error: No batch result files found in ${RUN_DIR}"
    echo "Please make sure 5_2_check_batch_new.sh has been run and batch jobs are completed"
    exit 1
fi

echo "Found ${#BATCH_RESULT_FILES[@]} batch result file(s)"
mkdir -p "${OUTPUT_DIR}"

# Process each batch result file
TOTAL_IMAGES=0
for batch_result_file in "${BATCH_RESULT_FILES[@]}"; do
    echo ""
    echo "Processing: $(basename ${batch_result_file})"
    
    # Check number of records
    RECORD_COUNT=$(python -c "
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
from src.dataset_generation.utils.json_utils import count_jsonl_records
print(count_jsonl_records('${batch_result_file}'))
")
    echo "  Records: ${RECORD_COUNT}"
    
    # Download images
    echo "  Downloading images..."
    python ${PROJECT_ROOT}/src/dataset_generation/batch_job/download_batch_job.py \
      --jsonl_path "${batch_result_file}" \
      --output_dir "${OUTPUT_DIR}"
    
    # Count downloaded images for this batch
    BATCH_IMAGE_COUNT=$(find "${OUTPUT_DIR}" -name "*.jpeg" -o -name "*.jpg" -o -name "*.png" | wc -l)
    echo "  ✓ Downloaded images (total so far: ${BATCH_IMAGE_COUNT})"
done

# Count total downloaded images
TOTAL_IMAGES=$(find "${OUTPUT_DIR}" -name "*.jpeg" -o -name "*.jpg" -o -name "*.png" | wc -l)

# Add image paths to scenarios for each merge source
echo ""
echo "Adding image paths to scenarios..."

# Find all texts_*.json files (one per merge source)
TEXT_FILES=("${ITERATE_DIR}/texts_by_merge_source"/texts_*.json)
if [ ${#TEXT_FILES[@]} -eq 0 ] || [ ! -f "${TEXT_FILES[0]}" ]; then
    # Fallback to root directory
    TEXT_FILES=("${ITERATE_DIR}"/texts_*.json)
fi

if [ ${#TEXT_FILES[@]} -gt 0 ] && [ -f "${TEXT_FILES[0]}" ]; then
    for text_file in "${TEXT_FILES[@]}"; do
        MERGE_SOURCE=$(basename "${text_file}" | sed 's/texts_\(.*\)\.json/\1/')
        OUTPUT_WITH_IMAGES="${ITERATE_DIR}/texts_with_images_${MERGE_SOURCE}.json"
        
        echo "  Processing merge source: ${MERGE_SOURCE}"
        
        python -c "
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
from src.dataset_generation.utils.img_utils import add_image_paths_to_scenarios
add_image_paths_to_scenarios(
    '${text_file}',
    '${OUTPUT_DIR}',
    '${OUTPUT_WITH_IMAGES}',
    '${MERGE_SOURCE}'
)
"
        echo "  ✓ Created: $(basename ${OUTPUT_WITH_IMAGES})"
    done
else
    echo "Warning: No text files found to add image paths"
fi

# Summary
echo ""
echo "✓ Image download and gather completed: ${ITERATE_DIR}"
echo "  - Downloaded ${TOTAL_IMAGES} images to: ${OUTPUT_DIR}"
echo "  - Processed ${#BATCH_RESULT_FILES[@]} batch result file(s)"
echo "  - Created texts_with_images_*.json for each merge source"
echo ""
echo "Next steps:"
echo "  - Use texts_with_images_*.json files for further processing"
echo "  - Review downloaded images in: ${OUTPUT_DIR}"

