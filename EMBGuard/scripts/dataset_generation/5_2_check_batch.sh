#!/bin/bash
# Step 5-2: Check Batch Job Status and Download Results - New Version
# Usage: Edit configuration below and run: ./scripts/5_2_check_batch_new.sh

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

OUTPUT_DIR="${BATCH_BASE_DIR}/${BATCH_RUN_TIMESTAMP}"

# ============================================
# Pipeline Execution
# ============================================

echo "Checking Batch Job Status: ${ITERATE_NAME}"
echo "Using timestamp: ${BATCH_RUN_TIMESTAMP}"
echo "Output directory: ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Find all batch job info files
JOB_INFO_FILES=("${OUTPUT_DIR}"/job_*.txt)
if [ ${#JOB_INFO_FILES[@]} -eq 0 ] || [ ! -f "${JOB_INFO_FILES[0]}" ]; then
    echo "Error: No batch job info files found in ${OUTPUT_DIR}"
    echo "Please make sure 5_1_text_to_image_w_batch_new.sh has been run"
    exit 1
fi

echo "Found ${#JOB_INFO_FILES[@]} batch job(s)"

# Extract batch job names from job info files and check each one
TOTAL_RECORDS=0
SUCCESSFUL_JOBS=0
FAILED_JOBS=0
SKIPPED_JOBS=0

for job_info_file in "${JOB_INFO_FILES[@]}"; do
    # Extract batch job name from job info file
    BATCH_JOB_NAME=$(grep "^Job ID:" "${job_info_file}" | cut -d' ' -f3 | tr -d '\r\n')
    
    if [ -z "${BATCH_JOB_NAME}" ]; then
        echo "Warning: Could not extract batch job name from ${job_info_file}"
        continue
    fi
    
    # Check if result file already exists
    RESULT_FILE="${OUTPUT_DIR}/batch_result_$(basename ${BATCH_JOB_NAME}).jsonl"
    
    if [ -f "${RESULT_FILE}" ]; then
        # Count records in existing result file
        RECORD_COUNT=$(python -c "
import json
count = 0
with open('${RESULT_FILE}', 'r') as f:
    for line in f:
        if line.strip():
            count += 1
print(count)
")
        TOTAL_RECORDS=$((TOTAL_RECORDS + RECORD_COUNT))
        SUCCESSFUL_JOBS=$((SUCCESSFUL_JOBS + 1))
        SKIPPED_JOBS=$((SKIPPED_JOBS + 1))
        echo ""
        echo "Skipping batch job (already downloaded): ${BATCH_JOB_NAME}"
        echo "  From: $(basename ${job_info_file})"
        echo "  ✓ Already exists: ${RECORD_COUNT} records"
        continue
    fi
    
    echo ""
    echo "Processing batch job: ${BATCH_JOB_NAME}"
    echo "  From: $(basename ${job_info_file})"
    
    # Check batch status and download results
    python ${PROJECT_ROOT}/src/dataset_generation/batch_job/check_batch.py \
      "${BATCH_JOB_NAME}" \
      --output_dir "${OUTPUT_DIR}"
    
    # Check if result file was downloaded
    if [ -f "${RESULT_FILE}" ]; then
        # Count records in result file
        RECORD_COUNT=$(python -c "
import json
count = 0
with open('${RESULT_FILE}', 'r') as f:
    for line in f:
        if line.strip():
            count += 1
print(count)
")
        TOTAL_RECORDS=$((TOTAL_RECORDS + RECORD_COUNT))
        SUCCESSFUL_JOBS=$((SUCCESSFUL_JOBS + 1))
        echo "  ✓ Downloaded ${RECORD_COUNT} records"
    else
        FAILED_JOBS=$((FAILED_JOBS + 1))
        echo "  ⚠️ No result file downloaded - batch may still be running"
    fi
done

# Summary
echo ""
echo "✓ Batch check completed: ${ITERATE_DIR}"
echo "  - Successful jobs: ${SUCCESSFUL_JOBS}/${#JOB_INFO_FILES[@]}"
echo "  - Skipped (already downloaded): ${SKIPPED_JOBS}"
echo "  - Failed/incomplete jobs: ${FAILED_JOBS}"
echo "  - Total records: ${TOTAL_RECORDS}"
echo ""
if [ ${SUCCESSFUL_JOBS} -gt 0 ]; then
    echo "Next steps:"
    echo "  - Run 5_3_download_image_new.sh to extract images from results"
    echo "  - Or use the result files directly for further processing"
fi

