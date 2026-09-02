#!/bin/bash
# Train Qwen3-VL with Full Fine-tuning using EMBHazard dataset
# All training configuration is in the YAML file
# Usage: bash scripts/train/train_qwen3vl_full.sh [YAML_FILE]

set -e

# ============================================
# Path settings
# ============================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLAMAFACTORY_DIR="${PROJECT_ROOT}/LlamaFactory"

# YAML file path (can be overridden by command line argument)
YAML_FILE="${1:-${PROJECT_ROOT}/scripts/train/qwen3vl_full_sft.yaml}"

# Set PYTHONPATH to include LlamaFactory src directory
export PYTHONPATH="${LLAMAFACTORY_DIR}/src:${PYTHONPATH}"


export WANDB_ENTITY="xxx"
export WANDB_PROJECT="xxx"

# ============================================
# Fix PyTorch/MKL library conflicts
# ============================================
# Workaround for "undefined symbol: iJIT_NotifyEvent" error
# This can occur when there are version conflicts between conda and pip PyTorch installations
# Try to use conda's MKL libraries first
if [ -n "${CONDA_PREFIX:-}" ]; then
    # Prefer conda's MKL libraries
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    # Disable Intel JIT profiling to avoid symbol conflicts
    export MKL_SERVICE_FORCE_INTEL=1
    export KMP_DUPLICATE_LIB_OK=TRUE
fi

# Use llamafactory-cli from conda environment if available
LLAMAFACTORY_CLI="xxxx"
if [ ! -f "${LLAMAFACTORY_CLI}" ]; then
    # Try to find llamafactory-cli in PATH
    LLAMAFACTORY_CLI=$(command -v llamafactory-cli 2>/dev/null || echo "")
fi

cd "${LLAMAFACTORY_DIR}"

# ============================================
# Validation
# ============================================

# Check if YAML file exists
if [ ! -f "${YAML_FILE}" ]; then
    echo "❌ Error: YAML file not found: ${YAML_FILE}"
    echo "   Usage: $0 [path/to/config.yaml]"
    exit 1
fi

echo "============================================================"
echo "Qwen3-VL Full Fine-tuning"
echo "============================================================"
echo "Config file: ${YAML_FILE}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "============================================================"
echo ""

# Extract dataset name from YAML to check dataset file
DATASET_NAME=$(grep -E "^dataset:" "${YAML_FILE}" | sed 's/.*dataset:[[:space:]]*\(.*\)/\1/' | sed 's/#.*$//' | xargs)
if [ -n "${DATASET_NAME}" ]; then
    # Handle multiple datasets (comma-separated)
    IFS=',' read -ra DATASETS <<< "${DATASET_NAME}"
    DATASET_INFO="${LLAMAFACTORY_DIR}/data/dataset_info.json"
    if [ -f "${DATASET_INFO}" ]; then
        for ds in "${DATASETS[@]}"; do
            ds=$(echo "$ds" | xargs)  # Trim whitespace
            DATASET_FILE=$(python3 -c "
import json
import sys
try:
    with open('${DATASET_INFO}', 'r') as f:
        data = json.load(f)
    if '${ds}' in data and 'file_name' in data['${ds}']:
        print(data['${ds}']['file_name'])
except:
    pass
" 2>/dev/null)
            
            if [ -n "${DATASET_FILE}" ] && [ -f "${DATASET_FILE}" ]; then
                echo "✓ Dataset file found: ${ds} -> ${DATASET_FILE}"
            elif [ -n "${DATASET_FILE}" ]; then
                echo "⚠️  Warning: Dataset file not found for ${ds}: ${DATASET_FILE}"
            fi
        done
    fi
fi



# ============================================
# Training execution
# ============================================

echo "Starting training with config: ${YAML_FILE}"
echo ""

PYTORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
echo "✓ PyTorch ${PYTORCH_VERSION} loaded successfully"
echo ""

echo "[DEBUG] WANDB_ENTITY=$WANDB_ENTITY"
echo "[DEBUG] WANDB_PROJECT=$WANDB_PROJECT"
env | grep -E '^WANDB_' | sort

# Use llamafactory-cli if available, otherwise use python module
if [ -n "${LLAMAFACTORY_CLI}" ] && [ -f "${LLAMAFACTORY_CLI}" ]; then
    "${LLAMAFACTORY_CLI}" train "${YAML_FILE}"
elif command -v llamafactory-cli &> /dev/null; then
    llamafactory-cli train "${YAML_FILE}"
else
    # Fallback: use python module
    python3 -m llamafactory.cli train "${YAML_FILE}"
fi

echo ""
echo "============================================================"
echo "✓ Training completed!"
echo "============================================================"

# Extract output directory from YAML
OUTPUT_DIR=$(grep -E "^output_dir:" "${YAML_FILE}" | sed 's/.*output_dir:[[:space:]]*\(.*\)/\1/' | sed 's/#.*$//' | xargs)
if [ -n "${OUTPUT_DIR}" ]; then
    if [[ "${OUTPUT_DIR}" == /* ]]; then
        echo "Output: ${OUTPUT_DIR}"
    else
        echo "Output: ${LLAMAFACTORY_DIR}/${OUTPUT_DIR}"
    fi
fi

# Display Wandb information
REPORT_TO=$(grep -E "^report_to:" "${YAML_FILE}" | sed 's/.*report_to:[[:space:]]*\(.*\)/\1/' | sed 's/#.*$//' | xargs)
if [ "${REPORT_TO}" = "wandb" ]; then
    RUN_NAME=$(grep -E "^run_name:" "${YAML_FILE}" | sed 's/.*run_name:[[:space:]]*\(.*\)/\1/' | sed 's/#.*$//' | xargs)
    WANDB_PROJECT=$(grep -E "^wandb_project:" "${YAML_FILE}" | sed 's/.*wandb_project:[[:space:]]*\(.*\)/\1/' | sed 's/#.*$//' | xargs)
    
    # Try to get project from environment variable if not in YAML
    if [ -z "${WANDB_PROJECT}" ]; then
        WANDB_PROJECT="${WANDB_PROJECT:-${WANDB_ENTITY:-}}"
    fi
    
    if [ -n "${WANDB_PROJECT}" ] && [ -n "${RUN_NAME}" ]; then
        echo "Wandb: https://wandb.ai/${WANDB_PROJECT}/${RUN_NAME}"
    elif [ -n "${RUN_NAME}" ]; then
        echo "Wandb run: ${RUN_NAME}"
        echo "  (Project will be determined by Wandb settings)"
    fi
fi
echo "============================================================"
