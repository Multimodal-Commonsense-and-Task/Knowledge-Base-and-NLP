#!/bin/bash
# Step 9: Construct Training Data for EMBGuard
# Converts downloaded CSV dataset (with images) to OpenAI format for multimodal SFT training
# Usage: Edit configuration below and run: ./scripts/dataset_generation/9_construct_train_data.sh

set -e  # Exit on any error

# ============================================
# Configuration - EDIT THESE
# ============================================

# Input CSV file path (downloaded dataset)
INPUT_CSV="/home/taeyoon/nas2/EMBGuardResults/EMBHazard_original_wo_filter_v1.0/train/dataset.csv"

# Output directory name (relative to RESULTS_DIR)
OUTPUT_DIR_NAME="EMBHazard"

# Iteration name (results folder name)
ITERATE_NAME="train"

# Output JSON file name (relative to iterate directory)
OUTPUT_FILE="embguard_train_data.json"

# Whether to include few-shot examples in training data
USE_FEW_SHOT=false

# ============================================
# Auto-configured paths
# ============================================

# Get project root (assuming script is in scripts/dataset_generation/ folder)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Output directory (EMBGuardResults on nas)
RESULTS_DIR="/home/taeyoon/nas2/EMBGuardResults"
OUTPUT_BASE_DIR="${RESULTS_DIR}/${OUTPUT_DIR_NAME}"
ITERATE_DIR="${OUTPUT_BASE_DIR}/${ITERATE_NAME}"

# Resolve input CSV path
if [[ "${INPUT_CSV}" == /* ]]; then
    # Absolute path
    INPUT_PATH="${INPUT_CSV}"
else
    # Relative to project root
    INPUT_PATH="${PROJECT_ROOT}/${INPUT_CSV}"
fi

# Resolve output path
if [[ "${OUTPUT_FILE}" == /* ]]; then
    OUTPUT_PATH="${OUTPUT_FILE}"
else
    OUTPUT_PATH="${ITERATE_DIR}/${OUTPUT_FILE}"
fi

# ============================================
# Validation
# ============================================

echo "EMBGuard Training Data Construction: ${ITERATE_NAME}"
echo "Input CSV: ${INPUT_PATH}"
echo "Output directory: ${OUTPUT_BASE_DIR}"
echo "Output JSON: ${OUTPUT_PATH}"
mkdir -p "${ITERATE_DIR}"

# Check if input CSV file exists
if [ ! -f "${INPUT_PATH}" ]; then
    echo "Error: Input CSV file not found: ${INPUT_PATH}"
    exit 1
fi

# Check if images directory exists (should be in same directory as CSV)
CSV_DIR="$(dirname "${INPUT_PATH}")"
IMAGES_DIR="${CSV_DIR}/images"
if [ ! -d "${IMAGES_DIR}" ]; then
    echo "Warning: Images directory not found: ${IMAGES_DIR}"
    echo "  Images may not be found during processing."
fi

# ============================================
# Pipeline Execution
# ============================================

echo ""
echo "Constructing training data in OpenAI format..."

# Build command arguments
USE_FEW_SHOT_ARG=""
if [ "${USE_FEW_SHOT}" = "false" ]; then
    USE_FEW_SHOT_ARG="--no-use-few-shot"
fi

# Note: model_name is required but not used for this task
# The task will load from CSV file and resolve image paths relative to CSV directory
python ${PROJECT_ROOT}/src/run_dataset_generation.py \
  --task embguard_train_data_construction \
  --model_name "dummy" \
  --data_path "${INPUT_PATH}" \
  --save_dir "${OUTPUT_BASE_DIR}" \
  --iterate_name "${ITERATE_NAME}" \
  ${USE_FEW_SHOT_ARG}

echo ""
echo "✓ Training data construction completed: ${ITERATE_DIR}"
echo "  - Output: ${OUTPUT_FILE}"
echo ""
echo "Next steps:"
echo "  1. Review the generated training data: ${OUTPUT_PATH}"
echo "  2. Upload to Hugging Face using upload_to_huggingface.sh"
echo "  3. Use for Qwen3-VL multimodal SFT training"
