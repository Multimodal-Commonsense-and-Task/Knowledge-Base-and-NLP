#!/bin/bash
# Merge LoRA adapter into base model for Qwen3-VL
# Usage: bash scripts/train/merge_adapter.sh [YAML_FILE] [OPTIONS]
#
# Simple usage (uses YAML file to auto-detect settings):
#   bash scripts/train/merge_adapter.sh scripts/train/train_qwen3vl_lora.yaml
#
# Merge and upload to Hugging Face:
#   bash scripts/train/merge_adapter.sh \
#     --upload \
#     --hf-repo-prefix qwen3-vl-4b-embhazard-merged
#
# Advanced usage (manual override):
#   bash scripts/train/merge_adapter.sh \
#     --yaml scripts/train/train_qwen3vl_lora.yaml \
#     --checkpoint checkpoint-1000 \
#     --upload \
#     --hf-repo-prefix my-model-merged

set -e

# ============================================
# Path settings
# ============================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLAMAFACTORY_DIR="${PROJECT_ROOT}/LlamaFactory"

# Set PYTHONPATH to include LlamaFactory src directory
export PYTHONPATH="${LLAMAFACTORY_DIR}/src:${PYTHONPATH}"

# ============================================
# Default configuration
# ============================================

# YAML file path (default: train_qwen3vl_lora.yaml)
YAML_FILE="${PROJECT_ROOT}/scripts/train/train_qwen3vl_lora.yaml"

# Base model (auto-detected from YAML)
BASE_MODEL="Qwen/Qwen3-VL-2B-Instruct"

# Adapter path (auto-detected from YAML output_dir)
ADAPTER_PATH="xxxx"

# Output directory (auto-generated: adapter_path + "_merged")
OUTPUT_DIR=""

# Checkpoint (optional, defaults to latest)
CHECKPOINT=""

# Torch dtype (default: auto)
TORCH_DTYPE="auto"

# Hugging Face upload settings
# Set to true to automatically upload merged models to Hugging Face
UPLOAD_TO_HF=false

# Hugging Face organization (default: EMBGuard)
HF_ORG="EMBGuard"

# Repository name prefix (auto-generates: {prefix}-cp{checkpoint_num})
# Example: "qwen3-vl-4b-embhazard-merged" → "qwen3-vl-4b-embhazard-merged-cp237"
# If empty, repo names are auto-generated from model paths
HF_REPO_PREFIX=""

# Make repositories private (true) or public (false)
HF_PRIVATE=true

# Hugging Face token (optional, can also set HF_TOKEN environment variable)
HF_TOKEN=""

# ============================================
# Parse command line arguments
# ============================================

# Check if first argument is a YAML file (without -- prefix)
if [ $# -gt 0 ] && [[ "$1" != --* ]] && [[ "$1" == *.yaml ]] || [[ "$1" == *.yml ]]; then
    YAML_FILE="$1"
    shift
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --yaml|--yaml_file|--config)
            YAML_FILE="$2"
            shift 2
            ;;
        --base_model_name_or_path|--base_model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --adapter_path|--adapter)
            ADAPTER_PATH="$2"
            shift 2
            ;;
        --output_dir|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --torch_dtype)
            TORCH_DTYPE="$2"
            shift 2
            ;;
        --upload|--upload-to-hf)
            UPLOAD_TO_HF=true
            shift
            ;;
        --no-upload)
            UPLOAD_TO_HF=false
            shift
            ;;
        --hf-org)
            HF_ORG="$2"
            shift 2
            ;;
        --hf-repo-prefix)
            HF_REPO_PREFIX="$2"
            shift 2
            ;;
        --hf-private)
            HF_PRIVATE=true
            shift
            ;;
        --hf-public)
            HF_PRIVATE=false
            shift
            ;;
        --hf-token)
            HF_TOKEN="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [YAML_FILE] [OPTIONS]"
            echo ""
            echo "Simple usage (auto-detect from YAML):"
            echo "  $0 scripts/train/train_qwen3vl_lora.yaml"
            echo "  $0 scripts/train/train_qwen3vl_lora.yaml --checkpoint checkpoint-1000"
            echo ""
            echo "Options:"
            echo "  YAML_FILE                                      YAML config file from training (default: train_qwen3vl_lora.yaml)"
            echo "  --yaml, --yaml_file, --config PATH            YAML config file path"
            echo "  --base_model_name_or_path, --base_model PATH  Override base model (auto from YAML)"
            echo "  --adapter_path, --adapter PATH                Override adapter path (auto from YAML output_dir)"
            echo "  --output_dir, --output PATH                   Override output directory (auto: adapter_path + '_merged')"
            echo "  --checkpoint CHECKPOINT                       Specific checkpoint to merge (e.g., 'checkpoint-1000')"
            echo "  --torch_dtype DTYPE                          Torch dtype: auto, float16, bfloat16, float32 (default: auto)"
            echo "  --upload, --upload-to-hf                      Upload merged models to Hugging Face after merge"
            echo "  --no-upload                                   Do not upload to Hugging Face (default)"
            echo "  --hf-org ORG                                  Hugging Face organization (default: EMBGuard)"
            echo "  --hf-repo-prefix PREFIX                      Prefix for repo names (auto-generates: {prefix}-cp{num})"
            echo "  --hf-private                                  Make repositories private (default)"
            echo "  --hf-public                                   Make repositories public"
            echo "  --hf-token TOKEN                             Hugging Face token (or set HF_TOKEN env var)"
            echo "  -h, --help                                    Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Simple: use default YAML"
            echo "  $0"
            echo ""
            echo "  # Use specific YAML file"
            echo "  $0 scripts/train/train_qwen3vl_lora.yaml"
            echo ""
            echo "  # Merge specific checkpoint"
            echo "  $0 --checkpoint checkpoint-1000"
            echo ""
            echo "  # Manual override"
            echo "  $0 --adapter_path /path/to/adapter --output_dir /path/to/merged"
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
# Auto-detect from YAML file
# ============================================

# Resolve YAML file path
if [[ "${YAML_FILE}" != /* ]]; then
    # Relative path - try relative to project root first
    if [ -f "${PROJECT_ROOT}/${YAML_FILE}" ]; then
        YAML_FILE="${PROJECT_ROOT}/${YAML_FILE}"
    elif [ -f "${YAML_FILE}" ]; then
        YAML_FILE="$(cd "$(dirname "${YAML_FILE}")" && pwd)/$(basename "${YAML_FILE}")"
    else
        echo "❌ Error: YAML file not found: ${YAML_FILE}"
        exit 1
    fi
fi

if [ ! -f "${YAML_FILE}" ]; then
    echo "❌ Error: YAML file not found: ${YAML_FILE}"
    exit 1
fi

echo "Reading configuration from: ${YAML_FILE}"

# Extract base model from YAML (if not manually set)
if [ -z "${BASE_MODEL}" ]; then
    BASE_MODEL=$(grep -E "^model_name_or_path:" "${YAML_FILE}" | sed 's/.*model_name_or_path:[[:space:]]*\(.*\)/\1/' | sed 's/#.*$//' | xargs)
    if [ -z "${BASE_MODEL}" ]; then
        BASE_MODEL="Qwen/Qwen3-VL-4B-Instruct"  # Default fallback
        echo "⚠️  Warning: Could not find model_name_or_path in YAML, using default: ${BASE_MODEL}"
    else
        echo "✓ Base model (from YAML): ${BASE_MODEL}"
    fi
fi

# Extract adapter path from YAML output_dir (if not manually set)
if [ -z "${ADAPTER_PATH}" ]; then
    ADAPTER_PATH=$(grep -E "^output_dir:" "${YAML_FILE}" | sed 's/.*output_dir:[[:space:]]*\(.*\)/\1/' | sed 's/#.*$//' | xargs)
    if [ -z "${ADAPTER_PATH}" ]; then
        echo "❌ Error: Could not find output_dir in YAML file: ${YAML_FILE}"
        echo "   Please specify --adapter_path manually"
        exit 1
    else
        echo "✓ Adapter path (from YAML): ${ADAPTER_PATH}"
    fi
fi

# Auto-generate output directory (if not manually set)
if [ -z "${OUTPUT_DIR}" ]; then
    OUTPUT_DIR="${ADAPTER_PATH}_merged"
    echo "✓ Output directory (auto-generated): ${OUTPUT_DIR}"
fi

# ============================================
# Validation
# ============================================

# Check if adapter path exists
if [ ! -d "${ADAPTER_PATH}" ]; then
    echo "❌ Error: Adapter path not found: ${ADAPTER_PATH}"
    echo "   Please check the output_dir in your YAML file or specify --adapter_path manually"
    exit 1
fi

# Find all checkpoints
ALL_CHECKPOINTS=()

# First, check if adapter files exist directly in adapter_path
HAS_DIRECT_ADAPTER=false
if [ -f "${ADAPTER_PATH}/adapter_model.bin" ] || [ -f "${ADAPTER_PATH}/adapter_model.safetensors" ]; then
    HAS_DIRECT_ADAPTER=true
    echo "✓ Found adapter files directly in: ${ADAPTER_PATH}"
fi

# Find all checkpoints (even if direct adapter exists)
while IFS= read -r checkpoint_dir; do
    checkpoint_name=$(basename "${checkpoint_dir}")
    # Check if checkpoint has adapter files
    if [ -f "${checkpoint_dir}/adapter_model.bin" ] || [ -f "${checkpoint_dir}/adapter_model.safetensors" ]; then
        ALL_CHECKPOINTS+=("${checkpoint_name}")
    fi
done < <(find "${ADAPTER_PATH}" -maxdepth 1 -type d -name "checkpoint-*" | sort -V)

# If direct adapter exists, add it to the list (empty string means use adapter_path directly)
if [ "${HAS_DIRECT_ADAPTER}" = "true" ]; then
    ALL_CHECKPOINTS=("" "${ALL_CHECKPOINTS[@]}")  # Add direct adapter first
fi

if [ ${#ALL_CHECKPOINTS[@]} -eq 0 ]; then
    echo "⚠️  Warning: No adapter files or checkpoints found in: ${ADAPTER_PATH}"
    echo "   Will attempt to load anyway..."
    ALL_CHECKPOINTS=("")  # Try anyway
else
    if [ "${HAS_DIRECT_ADAPTER}" = "true" ] && [ ${#ALL_CHECKPOINTS[@]} -gt 1 ]; then
        echo "✓ Found direct adapter + ${#ALL_CHECKPOINTS[@]} checkpoint(s) (total: ${#ALL_CHECKPOINTS[@]})"
    elif [ "${HAS_DIRECT_ADAPTER}" = "true" ]; then
        echo "✓ Found direct adapter files only"
    else
        echo "✓ Found ${#ALL_CHECKPOINTS[@]} checkpoint(s): ${ALL_CHECKPOINTS[*]}"
    fi
fi

# Determine which checkpoints to process
CHECKPOINTS_TO_MERGE=()
if [ -n "${CHECKPOINT}" ]; then
    # Specific checkpoint requested
    CHECKPOINT_PATH="${ADAPTER_PATH}/${CHECKPOINT}"
    if [ ! -d "${CHECKPOINT_PATH}" ]; then
        echo "❌ Error: Checkpoint not found: ${CHECKPOINT_PATH}"
        exit 1
    fi
    CHECKPOINTS_TO_MERGE=("${CHECKPOINT}")
    echo "✓ Will merge checkpoint: ${CHECKPOINT}"
else
    # No checkpoint specified - merge all
    CHECKPOINTS_TO_MERGE=("${ALL_CHECKPOINTS[@]}")
    if [ ${#CHECKPOINTS_TO_MERGE[@]} -gt 1 ]; then
        echo "✓ Will merge all ${#CHECKPOINTS_TO_MERGE[@]} checkpoint(s) automatically"
    elif [ ${#CHECKPOINTS_TO_MERGE[@]} -eq 1 ] && [ -n "${CHECKPOINTS_TO_MERGE[0]}" ]; then
        echo "✓ Will merge checkpoint: ${CHECKPOINTS_TO_MERGE[0]}"
    elif [ ${#CHECKPOINTS_TO_MERGE[@]} -eq 1 ] && [ -z "${CHECKPOINTS_TO_MERGE[0]}" ]; then
        echo "✓ Will merge adapter files directly"
    fi
fi

# ============================================
# Merge execution
# ============================================

# Build Python command
PYTHON_SCRIPT="${PROJECT_ROOT}/src/train/merge_adapter.py"

if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "❌ Error: Python script not found: ${PYTHON_SCRIPT}"
    exit 1
fi

# Process each checkpoint
TOTAL_CHECKPOINTS=${#CHECKPOINTS_TO_MERGE[@]}
CURRENT=0
MERGED_MODEL_PATHS=()  # Store paths of successfully merged models

for cp in "${CHECKPOINTS_TO_MERGE[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    # Determine output directory for this checkpoint
    if [ -z "${cp}" ]; then
        # Direct adapter files (no checkpoint subdirectory)
        CURRENT_OUTPUT_DIR="${OUTPUT_DIR}"
        CP_DISPLAY="adapter files"
    else
        # Checkpoint-specific output directory: {OUTPUT_DIR}_checkpoint-{number}
        CURRENT_OUTPUT_DIR="${OUTPUT_DIR}_${cp}"
        CP_DISPLAY="${cp}"
    fi
    
    echo "[INFO] Output directory for this checkpoint: ${CURRENT_OUTPUT_DIR}"
    
    echo ""
    echo "============================================================"
    echo "LoRA Adapter Merge [${CURRENT}/${TOTAL_CHECKPOINTS}]"
    echo "============================================================"
    echo "YAML config: ${YAML_FILE}"
    echo "Base model: ${BASE_MODEL}"
    echo "Adapter path: ${ADAPTER_PATH}"
    echo "Checkpoint: ${CP_DISPLAY}"
    echo "Output directory: ${CURRENT_OUTPUT_DIR}"
    echo "Torch dtype: ${TORCH_DTYPE}"
    echo "============================================================"
    echo ""
    
    # Build arguments
    PYTHON_ARGS=(
        "${PYTHON_SCRIPT}"
        --base_model_name_or_path "${BASE_MODEL}"
        --adapter_path "${ADAPTER_PATH}"
        --output_dir "${CURRENT_OUTPUT_DIR}"
        --torch_dtype "${TORCH_DTYPE}"
    )
    
    if [ -n "${cp}" ]; then
        PYTHON_ARGS+=(--checkpoint "${cp}")
    fi
    
    # Run merge script
    echo "Starting merge process for ${CP_DISPLAY}..."
    echo ""
    
    if python3 "${PYTHON_ARGS[@]}"; then
        echo ""
        echo "✓ Merge completed for ${CP_DISPLAY}!"
        echo "  Merged model saved to: ${CURRENT_OUTPUT_DIR}"
        # Store successfully merged model path
        MERGED_MODEL_PATHS+=("${CURRENT_OUTPUT_DIR}")
    else
        echo ""
        echo "❌ Error: Merge failed for ${CP_DISPLAY}"
        echo "  Continuing with next checkpoint..."
    fi
done

echo ""
echo "============================================================"
echo "✓ All merges completed!"
echo "============================================================"
if [ ${TOTAL_CHECKPOINTS} -eq 1 ]; then
    if [ -z "${CHECKPOINTS_TO_MERGE[0]}" ]; then
        echo "Merged model saved to: ${OUTPUT_DIR}"
    else
        echo "Merged model saved to: ${OUTPUT_DIR}_${CHECKPOINTS_TO_MERGE[0]}"
    fi
else
    echo "Merged ${TOTAL_CHECKPOINTS} checkpoint(s) to separate directories:"
    for cp in "${CHECKPOINTS_TO_MERGE[@]}"; do
        if [ -z "${cp}" ]; then
            echo "  ✓ ${OUTPUT_DIR}"
        else
            echo "  ✓ ${OUTPUT_DIR}_${cp}"
        fi
    done
fi
echo "============================================================"

# ============================================
# Upload to Hugging Face (if requested)
# ============================================

if [ "${UPLOAD_TO_HF}" = "true" ] && [ ${#MERGED_MODEL_PATHS[@]} -gt 0 ]; then
    echo ""
    echo "============================================================"
    echo "Uploading Merged Models to Hugging Face"
    echo "============================================================"
    echo "Organization: ${HF_ORG}"
    echo "Private: ${HF_PRIVATE}"
    echo "Models to upload: ${#MERGED_MODEL_PATHS[@]}"
    echo ""
    
    # Get Hugging Face token
    if [ -z "${HF_TOKEN}" ]; then
        # Try environment variables
        if [ -n "${HF_TOKEN:-}" ]; then
            HF_TOKEN="${HF_TOKEN}"
        elif [ -n "${HUGGINGFACE_TOKEN:-}" ]; then
            HF_TOKEN="${HUGGINGFACE_TOKEN}"
        fi
    fi
    
    # Build Python command
    UPLOAD_SCRIPT="${PROJECT_ROOT}/src/hf_utils/upload_merged_models.py"
    
    if [ ! -f "${UPLOAD_SCRIPT}" ]; then
        echo "❌ Error: Upload script not found: ${UPLOAD_SCRIPT}"
        echo "   Skipping upload..."
    else
        # Build arguments
        UPLOAD_ARGS=(
            "${UPLOAD_SCRIPT}"
            --org "${HF_ORG}"
            --models "${MERGED_MODEL_PATHS[@]}"
        )
        
        # Set repo prefix if provided, otherwise auto-generate from model paths
        if [ -n "${HF_REPO_PREFIX}" ]; then
            UPLOAD_ARGS+=(--repo-prefix "${HF_REPO_PREFIX}")
        fi
        
        if [ "${HF_PRIVATE}" = "true" ]; then
            UPLOAD_ARGS+=(--private)
        fi
        
        if [ -n "${HF_TOKEN}" ]; then
            UPLOAD_ARGS+=(--token "${HF_TOKEN}")
        fi
        
        # Execute upload
        echo "Starting upload process..."
        echo ""
        
        if python3 "${UPLOAD_ARGS[@]}"; then
            echo ""
            echo "============================================================"
            echo "✓ Upload completed!"
            echo "============================================================"
        else
            echo ""
            echo "============================================================"
            echo "⚠️  Warning: Upload failed or was cancelled"
            echo "============================================================"
        fi
    fi
elif [ "${UPLOAD_TO_HF}" = "true" ] && [ ${#MERGED_MODEL_PATHS[@]} -eq 0 ]; then
    echo ""
    echo "⚠️  Warning: Upload requested but no models were successfully merged"
fi
