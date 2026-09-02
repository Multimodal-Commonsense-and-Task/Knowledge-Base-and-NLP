#!/bin/bash
# Cancel batch jobs created by 5_1_text_to_image_w_batch.sh
# Usage: 
#   ./5_1_cancel_batch.sh --iterate_name train
#   ./5_1_cancel_batch.sh --batch_job batches/xxx
#   ./5_1_cancel_batch.sh --timestamp 1234567890

set -e  # Exit on any error

# ============================================
# Configuration - EDIT THESE
# ============================================

# Iteration name (results folder name)
# Leave empty to use command line argument
ITERATE_NAME="train"

# Specific batch job ID (e.g., batches/xxx)
# Leave empty to use command line argument
BATCH_JOB=""

# Timestamp to find batch jobs
# Leave empty to use command line argument
TIMESTAMP=""

# Dry run mode (show what would be cancelled without actually cancelling)
DRY_RUN=false

# ============================================
# Auto-configured paths
# ============================================

# Get project root (assuming script is in scripts/dataset_generation/ folder)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# ============================================
# Build command arguments
# ============================================

ARGS=()

if [ -n "${ITERATE_NAME}" ]; then
    ARGS+=("--iterate_name" "${ITERATE_NAME}")
elif [ -n "${BATCH_JOB}" ]; then
    ARGS+=("--batch_job" "${BATCH_JOB}")
elif [ -n "${TIMESTAMP}" ]; then
    ARGS+=("--timestamp" "${TIMESTAMP}")
fi

# Add command line arguments (override config)
if [ $# -gt 0 ]; then
    ARGS=("$@")
fi

if [ "${DRY_RUN}" = "true" ]; then
    ARGS+=("--dry_run")
fi

# ============================================
# Execute cancellation
# ============================================

echo "Cancelling batch jobs..."
echo "Project root: ${PROJECT_ROOT}"
echo ""

python ${PROJECT_ROOT}/src/dataset_generation/batch_job/cancel_batch.py "${ARGS[@]}"

echo ""
echo "✓ Done"
