#!/bin/bash
# Upload merged model directories to Hugging Face Hub
# 
# Usage:
#   1. Edit configuration below and run: bash scripts/train/upload_merged_models.sh
#   2. Or use command line arguments: bash scripts/train/upload_merged_models.sh [OPTIONS]
#
# Example (script configuration):
#   Edit MODELS array below, then run:
#     bash scripts/train/upload_merged_models.sh
#
# Example (command line):
#   bash scripts/train/upload_merged_models.sh \
#     --models /path/to/merged1 /path/to/merged2 \
#     --repo-prefix qwen3-vl-4b-embhazard-merged

set -e

# ============================================
# Path settings
# ============================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

# ============================================
# Configuration - EDIT THESE
# ============================================

# Hugging Face organization name
HF_ORG="EMBGuard"

# Model directories to upload (space-separated paths)
# Edit this array with your merged model paths
MODELS=(
    "saves/EMBGuard/train/qwen3-vl-2b_language_only/checkpoint-711"
)

# Repository name prefix (auto-generates: {prefix}-cp{checkpoint_num})
# Example: "qwen3-vl-4b-embhazard-merged" → "qwen3-vl-4b-embhazard-merged-cp237"
# If empty, repo names are auto-generated from model paths
# Example:
#   REPO_PREFIX="qwen3-vl-4b-embhazard-merged"
REPO_PREFIX=""

# Custom repository names (optional, must match number of models if provided)
# If set, overrides REPO_PREFIX
REPO_NAMES=(
    "EMBGuard-2B"
)

# Make repositories private (true) or public (false)
PRIVATE=false

# Hugging Face token (optional, can also set HF_TOKEN environment variable)
# HF_TOKEN="your_token_here"
HF_TOKEN="xxxx"

# ============================================
# Parse command line arguments
# ============================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --org)
            HF_ORG="$2"
            shift 2
            ;;
        --models)
            shift
            # Collect all model paths until next option
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                MODELS+=("$1")
                shift
            done
            ;;
        --repo-prefix)
            REPO_PREFIX="$2"
            shift 2
            ;;
        --repo-names)
            shift
            # Collect all repo names until next option
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                REPO_NAMES+=("$1")
                shift
            done
            ;;
        --private)
            PRIVATE="true"
            shift
            ;;
        --public)
            PRIVATE="false"
            shift
            ;;
        --token)
            HF_TOKEN="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --org ORG_NAME                    Hugging Face organization name (default: EMBGuard)"
            echo "  --models PATH1 PATH2 ...          Model directories to upload (required)"
            echo "  --repo-prefix PREFIX             Prefix for auto-generated repo names"
            echo "  --repo-names NAME1 NAME2 ...     Custom repository names (must match --models count)"
            echo "  --private                         Make repositories private (default)"
            echo "  --public                          Make repositories public"
            echo "  --token TOKEN                     Hugging Face token (or set HF_TOKEN env var)"
            echo "  -h, --help                        Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Upload with auto-generated repo names"
            echo "  $0 \\"
            echo "    --org EMBGuard \\"
            echo "    --models \\"
            echo "      /path/to/merged_checkpoint-1000 \\"
            echo "      /path/to/merged_checkpoint-2000 \\"
            echo "    --repo-prefix qwen3-vl-4b-embhazard-merged"
            echo ""
            echo "  # Upload with custom repo names"
            echo "  $0 \\"
            echo "    --org EMBGuard \\"
            echo "    --models \\"
            echo "      /path/to/merged_checkpoint-1000 \\"
            echo "      /path/to/merged_checkpoint-2000 \\"
            echo "    --repo-names \\"
            echo "      qwen3-vl-4b-embhazard-merged-cp1000 \\"
            echo "      qwen3-vl-4b-embhazard-merged-cp2000"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================
# Override with command line arguments (if provided)
# ============================================

# Command line arguments can override the settings above
# This allows both script-based and command-line configuration

# ============================================
# Validation
# ============================================

# If MODELS is empty and no command line arguments provided, show error
if [ ${#MODELS[@]} -eq 0 ] && [ $# -eq 0 ]; then
    echo "❌ Error: No models specified"
    echo ""
    echo "Please either:"
    echo "  1. Edit MODELS array in this script (lines ~35-40)"
    echo "  2. Use --models command line argument"
    echo ""
    echo "Use --help for usage information"
    exit 1
fi

# Check if model directories exist
for model_path in "${MODELS[@]}"; do
    if [ ! -d "${model_path}" ]; then
        echo "❌ Error: Model directory not found: ${model_path}"
        exit 1
    fi
done

# Validate repo names count if provided
if [ ${#REPO_NAMES[@]} -gt 0 ] && [ ${#REPO_NAMES[@]} -ne ${#MODELS[@]} ]; then
    echo "❌ Error: Number of --repo-names (${#REPO_NAMES[@]}) does not match number of --models (${#MODELS[@]})"
    exit 1
fi

# Get Hugging Face token
if [ -z "${HF_TOKEN}" ]; then
    # Try environment variables
    if [ -n "${HF_TOKEN:-}" ]; then
        HF_TOKEN="${HF_TOKEN}"
    elif [ -n "${HUGGINGFACE_TOKEN:-}" ]; then
        HF_TOKEN="${HUGGINGFACE_TOKEN}"
    fi
fi

# ============================================
# Display configuration
# ============================================

echo "============================================================"
echo "Upload Merged Models to Hugging Face"
echo "============================================================"
echo "Organization: ${HF_ORG}"
echo "Private: ${PRIVATE}"
echo "Models to upload: ${#MODELS[@]}"
echo ""
for i in "${!MODELS[@]}"; do
    model_path="${MODELS[$i]}"
    if [ ${#REPO_NAMES[@]} -gt 0 ]; then
        repo_name="${REPO_NAMES[$i]}"
    elif [ -n "${REPO_PREFIX}" ]; then
        # Extract checkpoint number if present
        model_name=$(basename "${model_path}")
        if [[ "${model_name}" == *"checkpoint-"* ]]; then
            checkpoint_num=$(echo "${model_name}" | sed 's/.*checkpoint-\([0-9]*\).*/\1/')
            repo_name="${REPO_PREFIX}-cp${checkpoint_num}"
        else
            repo_name="${REPO_PREFIX}"
        fi
    else
        repo_name="(auto-generated)"
    fi
    echo "  [$(($i + 1))] $(basename "${model_path}")"
    echo "      → ${HF_ORG}/${repo_name}"
done
echo "============================================================"
echo ""

# ============================================
# Build Python command
# ============================================

PYTHON_SCRIPT="${PROJECT_ROOT}/src/hf_utils/upload_merged_models.py"

if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "❌ Error: Python script not found: ${PYTHON_SCRIPT}"
    exit 1
fi

# Build arguments
PYTHON_ARGS=(
    "${PYTHON_SCRIPT}"
    --org "${HF_ORG}"
    --models "${MODELS[@]}"
)

if [ -n "${REPO_PREFIX}" ]; then
    PYTHON_ARGS+=(--repo-prefix "${REPO_PREFIX}")
elif [ ${#REPO_NAMES[@]} -gt 0 ]; then
    PYTHON_ARGS+=(--repo-names "${REPO_NAMES[@]}")
fi

if [ "${PRIVATE}" = "true" ]; then
    PYTHON_ARGS+=(--private)
fi

if [ -n "${HF_TOKEN}" ]; then
    PYTHON_ARGS+=(--token "${HF_TOKEN}")
fi

# ============================================
# Execute upload
# ============================================

echo "Starting upload process..."
echo ""

python3 "${PYTHON_ARGS[@]}"

echo ""
echo "============================================================"
echo "✓ Upload process completed!"
echo "============================================================"
