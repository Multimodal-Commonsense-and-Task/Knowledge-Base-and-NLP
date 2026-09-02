#!/bin/bash
# Download EMBHazard dataset from Hugging Face with images
# Downloads the dataset and saves as original files (CSV + image files)
# Usage: Edit configuration below and run: ./scripts/download_EMBHazard_dataset.sh

set -e  # Exit on any error

# ============================================
# Configuration - EDIT THESE
# ============================================

# Hugging Face dataset ID
DATASET_ID="EMBGuard/EMBHazard_original_wo_filter_v1.0"

# Dataset split to download (leave empty to download all splits)
# Examples: "train", "validation", "test", or "" for all splits
DATASET_SPLIT="train"

# Output directory (where to save the dataset)
OUTPUT_DIR="/home/taeyoon/nas2/EMBGuardResults/EMBHazard_original_wo_filter_v1.0"

# Hugging Face cache directory (optional, uses default if not set)
HF_CACHE_DIR="/home/taeyoon/nas2/huggingface_cache"

# ============================================
# Auto-configured paths
# ============================================

# Get project root (assuming script is in scripts/ folder)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# ============================================
# Get Hugging Face Token
# ============================================
# Priority: command line argument > HF_TOKEN env var > HUGGINGFACE_TOKEN env var
HF_TOKEN=""

if [ -n "${1:-}" ]; then
    # Token provided as first command line argument
    HF_TOKEN="${1}"
elif [ -n "${HF_TOKEN:-}" ]; then
    # Token from HF_TOKEN environment variable
    HF_TOKEN="${HF_TOKEN}"
elif [ -n "${HUGGINGFACE_TOKEN:-}" ]; then
    # Token from HUGGINGFACE_TOKEN environment variable
    HF_TOKEN="${HUGGINGFACE_TOKEN}"
fi

# ============================================
# Set Environment Variables
# ============================================

# Set cache directory if specified
if [ -n "${HF_CACHE_DIR}" ]; then
    export HF_HOME="${HF_CACHE_DIR}"
    export HF_DATASETS_CACHE="${HF_CACHE_DIR}/datasets"
    echo "Using Hugging Face cache: ${HF_CACHE_DIR}"
fi

# ============================================
# Download Dataset
# ============================================

echo "Downloading EMBHazard Dataset"
echo "Dataset ID: ${DATASET_ID}"
echo "Split: ${DATASET_SPLIT:-all}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Build command arguments
SPLIT_ARG=""
if [ -n "${DATASET_SPLIT}" ]; then
    SPLIT_ARG="--split ${DATASET_SPLIT}"
fi

TOKEN_ARG=""
if [ -n "${HF_TOKEN}" ]; then
    TOKEN_ARG="--token ${HF_TOKEN}"
    echo "Using Hugging Face token for authentication."
else
    echo "Warning: No Hugging Face token provided."
    echo "  This may fail for private datasets."
    echo "  Usage:"
    echo "    export HF_TOKEN='your-token'"
    echo "    bash $0"
    echo "  Or:"
    echo "    bash $0 'your-token'"
    echo ""
fi

# Run download script
python -m src.hf_utils.download_hf_dataset \
    --dataset-id "${DATASET_ID}" \
    --output-dir "${OUTPUT_DIR}" \
    ${SPLIT_ARG} \
    ${TOKEN_ARG}

echo ""
echo "✓ Dataset download completed"
echo "  Output directory: ${OUTPUT_DIR}"
if [ -n "${DATASET_SPLIT}" ]; then
    echo "  Split: ${DATASET_SPLIT}"
    echo "  CSV file: ${OUTPUT_DIR}/${DATASET_SPLIT}/dataset.csv"
    echo "  Images: ${OUTPUT_DIR}/${DATASET_SPLIT}/images/"
else
    echo "  All splits downloaded"
fi
