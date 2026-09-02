#!/bin/bash
# Step 5-3: Download Images from Single Batch Input File - New Version
# Usage: Edit configuration below and run: ./scripts/5_3_download_image_single.sh
# This script downloads images for a single batch input file

set -e  # Exit on any error

# ============================================
# Configuration - EDIT THESE
# ============================================

# Iteration name (results folder name)
ITERATE_NAME="train"

# Batch run timestamp (directory under raw/batch/)
# Leave empty to auto-detect the latest timestamp
BATCH_RUN_TIMESTAMP="1768294840"

# Batch input file name (e.g., "batch_input_1768294906.jsonl")
# This should match one of the batch_input_*.jsonl files in the batch run directory
BATCH_INPUT_FILE="batch_input_1768294906.jsonl"

# Output subdirectory (optional, for organizing downloaded images)
# Leave empty to use default: images/downloaded
OUTPUT_SUBDIR=""

# ============================================
# Auto-configured paths
# ============================================

# Get project root (assuming script is in scripts/dataset_generation/ folder)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

RESULTS_DIR="${PROJECT_ROOT}/dataset_generation_output"
ITERATE_DIR="${RESULTS_DIR}/${ITERATE_NAME}"
BATCH_BASE_DIR="${ITERATE_DIR}/raw/batch"

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

# Determine output directory
if [ -n "${OUTPUT_SUBDIR}" ]; then
    OUTPUT_DIR="${ITERATE_DIR}/images/downloaded/${OUTPUT_SUBDIR}"
else
    # Extract merge_source from batch input filename if possible, or use default
    OUTPUT_DIR="${ITERATE_DIR}/images/downloaded"
fi

# ============================================
# Validation
# ============================================

echo "Downloading Images from Single Batch Input: ${ITERATE_NAME}"
echo "Using timestamp: ${BATCH_RUN_TIMESTAMP}"
echo "Batch input file: ${BATCH_INPUT_FILE}"
echo "Output directory: ${OUTPUT_DIR}"

# Check if batch input file exists
BATCH_INPUT_PATH="${RUN_DIR}/${BATCH_INPUT_FILE}"
if [ ! -f "${BATCH_INPUT_PATH}" ]; then
    echo "Error: Batch input file not found: ${BATCH_INPUT_PATH}"
    echo "Available batch input files:"
    ls -1 "${RUN_DIR}"/batch_input_*.jsonl 2>/dev/null | xargs -n1 basename || echo "  (none found)"
    exit 1
fi

# Find corresponding batch result file
# Extract timestamp from batch_input filename (e.g., batch_input_1768294906.jsonl -> 1768294906)
BATCH_TIMESTAMP=$(echo "${BATCH_INPUT_FILE}" | sed -n 's/batch_input_\([0-9]*\)\.jsonl/\1/p')
if [ -z "${BATCH_TIMESTAMP}" ]; then
    echo "Error: Could not extract timestamp from batch input filename: ${BATCH_INPUT_FILE}"
    echo "Expected format: batch_input_<timestamp>.jsonl"
    exit 1
fi

# Try to find the corresponding batch result file
# First, check if there's a job file with the same timestamp
JOB_FILE="${RUN_DIR}/job_${BATCH_TIMESTAMP}.txt"
BATCH_RESULT_FILE=""

if [ -f "${JOB_FILE}" ]; then
    # Extract Job Short ID from job file
    JOB_SHORT_ID=$(python -c "
try:
    with open('${JOB_FILE}', 'r') as f:
        for line in f:
            if 'Job Short ID:' in line:
                # Extract the ID after 'Job Short ID:'
                parts = line.split('Job Short ID:', 1)
                if len(parts) > 1:
                    job_id = parts[1].strip()
                    print(job_id)
                    break
except:
    pass
" 2>/dev/null)
    
    if [ -n "${JOB_SHORT_ID}" ]; then
        # Look for batch_result file with this job ID
        POTENTIAL_RESULT="${RUN_DIR}/batch_result_${JOB_SHORT_ID}.jsonl"
        if [ -f "${POTENTIAL_RESULT}" ]; then
            BATCH_RESULT_FILE="${POTENTIAL_RESULT}"
            echo "Found corresponding batch result file: $(basename ${BATCH_RESULT_FILE})"
        fi
    fi
fi

# If not found, try to find by pattern matching (batch_result files are named by job ID)
if [ -z "${BATCH_RESULT_FILE}" ]; then
    echo "Warning: Could not automatically find batch result file for ${BATCH_INPUT_FILE}"
    echo "Available batch result files:"
    ls -1 "${RUN_DIR}"/batch_result_*.jsonl 2>/dev/null | xargs -n1 basename || echo "  (none found)"
    echo ""
    echo "Please make sure:"
    echo "  1. The job file exists: ${JOB_FILE}"
    echo "  2. 5_2_check_batch.sh has been run to download batch results"
    echo "  3. The batch job is completed"
    exit 1
fi

# ============================================
# Pipeline Execution
# ============================================

echo ""
echo "Processing batch input: ${BATCH_INPUT_FILE}"
echo "Corresponding result file: $(basename ${BATCH_RESULT_FILE})"

# Check number of records
RECORD_COUNT=$(python -c "
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
from src.dataset_generation.utils.json_utils import count_jsonl_records
print(count_jsonl_records('${BATCH_RESULT_FILE}'))
")
echo "  Records: ${RECORD_COUNT}"

# Download images
echo "  Downloading images..."
mkdir -p "${OUTPUT_DIR}"
python ${PROJECT_ROOT}/src/dataset_generation/batch_job/download_batch_job.py \
  --jsonl_path "${BATCH_RESULT_FILE}" \
  --output_dir "${OUTPUT_DIR}"

# Count downloaded images
TOTAL_IMAGES=$(find "${OUTPUT_DIR}" -name "*.jpeg" -o -name "*.jpg" -o -name "*.png" | wc -l)
echo "  ✓ Downloaded ${TOTAL_IMAGES} images to: ${OUTPUT_DIR}"

# Optionally add image paths to scenarios
# Extract merge_source from the batch input file if possible
echo ""
echo "Checking for corresponding text file to add image paths..."

# Try to find the corresponding text file
# The merge_source can be inferred from the batch input file's custom_id pattern
MERGE_SOURCE=$(python -c "
import json
import sys
try:
    with open('${BATCH_INPUT_PATH}', 'r') as f:
        first_line = f.readline()
        if first_line.strip():
            data = json.loads(first_line)
            custom_id = data.get('custom_id', '')
            if '-' in custom_id:
                merge_source = custom_id.split('-', 1)[1]
                print(merge_source)
            else:
                print('')
        else:
            print('')
except:
    print('')
" 2>/dev/null)

if [ -n "${MERGE_SOURCE}" ]; then
    TEXT_FILE="${ITERATE_DIR}/texts_by_merge_source/texts_${MERGE_SOURCE}.json"
    if [ -f "${TEXT_FILE}" ]; then
        OUTPUT_WITH_IMAGES="${ITERATE_DIR}/texts_with_images_${MERGE_SOURCE}.json"
        
        echo "  Found merge source: ${MERGE_SOURCE}"
        echo "  Adding image paths to scenarios..."
        
        python -c "
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
from src.dataset_generation.utils.img_utils import add_image_paths_to_scenarios
add_image_paths_to_scenarios(
    '${TEXT_FILE}',
    '${OUTPUT_DIR}',
    '${OUTPUT_WITH_IMAGES}',
    '${MERGE_SOURCE}'
)
"
        echo "  ✓ Created: $(basename ${OUTPUT_WITH_IMAGES})"
    else
        echo "  Note: Text file not found: ${TEXT_FILE}"
        echo "  Skipping image path addition to scenarios"
    fi
else
    echo "  Note: Could not determine merge_source from batch input file"
    echo "  Skipping image path addition to scenarios"
fi

# Summary
echo ""
echo "✓ Image download completed: ${ITERATE_DIR}"
echo "  - Downloaded ${TOTAL_IMAGES} images to: ${OUTPUT_DIR}"
echo "  - Processed batch input: ${BATCH_INPUT_FILE}"
echo "  - Result file: $(basename ${BATCH_RESULT_FILE})"
if [ -n "${MERGE_SOURCE}" ] && [ -f "${OUTPUT_WITH_IMAGES}" ]; then
    echo "  - Created: texts_with_images_${MERGE_SOURCE}.json"
fi
