#!/bin/bash

# EMBGuard Hugging Face Upload Script
# Usage: bash scripts/upload_to_huggingface.sh
# Edit the configuration below to change upload settings

# Get project root (assuming script is in scripts/ folder)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# ============================================
# Configuration - Edit these values as needed
# ============================================
# Hugging Face organization name
HF_ORG="EMBGuard"

# Option 1: Use predefined datasets
# Which dataset(s) to upload: "EMBGuardTest", "heldout_set", or "both"
# Leave empty to use custom file (Option 2 or 3)
DATASET=""

# EMBGuardTest split names: "full" -> causal_risky/etc., "legacy" -> HR/HNR/MHR/NHR
TEST_SPLIT_NAMES="full"

# Dataset schema: "v3" -> normalized public columns, "legacy" -> original CSV columns
SCHEMA="v3"

# Option 2: Upload custom CSV file
# Path to CSV file (relative to project root or absolute path)
#CSV_PATH="xxx"

# Option 3: Upload custom JSON file (OpenAI format training data)
# Path to JSON file (relative to project root or absolute path)
JSON_PATH="xxx"

# Dataset name for custom file (required if CSV_PATH or JSON_PATH is set)
DATASET_NAME="EMBHazard"

# Base directory for resolving image paths (optional, default: project root)
# If image paths are relative to project root, leave this empty
# If image paths are relative to a specific directory, set it here
# BASE_DIR="dataset_generation_output/train/images/downloaded"

# Make dataset private (set to "true" to make private)
PRIVATE="false"

# Build the dataset and print split/schema information without uploading
DRY_RUN="false"

# Hugging Face token
# Option 1: Set it here directly (not recommended for security)
# HF_TOKEN="your_token_here"
# Option 2: Set HF_TOKEN environment variable before running the script
  export HF_TOKEN="xxxx"
# Option 3: Use huggingface-cli login (recommended)
#   huggingface-cli login
# ============================================

# Check if token is available
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN not set. Trying to use huggingface-cli login..."
    if command -v huggingface-cli &> /dev/null; then
        echo "  Found huggingface-cli. If you're already logged in, this should work."
    else
        echo "  Error: HF_TOKEN environment variable is required."
        echo "  Please set it by:"
        echo "    1. export HF_TOKEN=\"your_token_here\""
        echo "    2. Or uncomment and set HF_TOKEN in this script"
        echo "    3. Or run: huggingface-cli login"
        exit 1
    fi
fi

echo "============================================================"
echo "EMBGuard Hugging Face Upload"
echo "============================================================"
echo "Organization: $HF_ORG"
echo "Private: $PRIVATE"
echo "Schema: $SCHEMA"
if [ -n "$DATASET" ]; then
    echo "Dataset: $DATASET (predefined)"
elif [ -n "$JSON_PATH" ]; then
    echo "JSON Path: $JSON_PATH"
    echo "Dataset Name: $DATASET_NAME"
    if [ -n "$BASE_DIR" ]; then
        echo "Base Dir: $BASE_DIR"
    fi
elif [ -n "$CSV_PATH" ]; then
    echo "CSV Path: $CSV_PATH"
    echo "Dataset Name: $DATASET_NAME"
    if [ -n "$BASE_DIR" ]; then
        echo "Base Dir: $BASE_DIR"
    fi
else
    echo "Error: Must specify DATASET, CSV_PATH, or JSON_PATH"
    exit 1
fi
echo "============================================================"
echo ""

# Build command arguments
CMD_ARGS=(
    "--org" "$HF_ORG"
)

if [ -n "$DATASET" ]; then
    # Use predefined dataset
    CMD_ARGS+=("--dataset" "$DATASET")
    CMD_ARGS+=("--test-split-names" "$TEST_SPLIT_NAMES")
elif [ -n "$JSON_PATH" ]; then
    # Use custom JSON
    if [ -z "$DATASET_NAME" ]; then
        echo "Error: DATASET_NAME must be set when JSON_PATH is set"
        exit 1
    fi
    CMD_ARGS+=("--json-path" "$JSON_PATH")
    CMD_ARGS+=("--dataset-name" "$DATASET_NAME")
    if [ -n "$BASE_DIR" ]; then
        CMD_ARGS+=("--base-dir" "$BASE_DIR")
    fi
elif [ -n "$CSV_PATH" ]; then
    # Use custom CSV
    if [ -z "$DATASET_NAME" ]; then
        echo "Error: DATASET_NAME must be set when CSV_PATH is set"
        exit 1
    fi
    CMD_ARGS+=("--csv-path" "$CSV_PATH")
    CMD_ARGS+=("--dataset-name" "$DATASET_NAME")
    if [ -n "$BASE_DIR" ]; then
        CMD_ARGS+=("--base-dir" "$BASE_DIR")
    fi
else
    echo "Error: Must specify DATASET, CSV_PATH, or JSON_PATH"
    exit 1
fi

CMD_ARGS+=("--schema" "$SCHEMA")

if [ "$PRIVATE" = "true" ]; then
    CMD_ARGS+=("--private")
fi

if [ "$DRY_RUN" = "true" ]; then
    CMD_ARGS+=("--dry-run")
fi

if [ -n "$HF_TOKEN" ]; then
    CMD_ARGS+=("--token" "$HF_TOKEN")
fi

# Execute command
python -m src.hf_utils.upload_to_huggingface "${CMD_ARGS[@]}"
