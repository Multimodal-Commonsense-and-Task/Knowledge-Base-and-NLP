#!/bin/bash
# Make dataset CSV from scenario JSON(s) - New Version
#
# Output columns:
# Category,Subcategory,Type,ID,Situation,Action,Risk,Risk Type,Related Hazard,Mitigate Action,URL
#
# Mapping:
# - Category        := risk_type
# - Subcategory     := mechanism
# - Type            := type
# - ID              := id (dual scenarios become id_v1/id_v2)
# - Situation       := situation (or situation1/2 if present)
# - Action          := action (or action1/2 for dual)
# - Risk            := O
# - Risk Type       := Category
# - Related Hazard  := hazard (or hazard1/2 for dual)
# - Mitigate Action := X
# - URL             := url
#
# Usage:
# - Defaults: ./scripts/7_make_dataset_new.sh
# - Custom inputs/outputs:
#   ./scripts/7_make_dataset_new.sh --inputs <a.json> [b.json ...] --output <out.csv> [--require-url|--no-require-url]

set -e

# ============================================
# Configuration - EDIT THESE
# ============================================

DEFAULT_ITERATE_NAME="train"

# Input files will be auto-detected (texts_with_images_*.json)
# Or specify manually using --inputs or --input-glob

# Output CSV path (relative to results/${ITERATE_NAME})
DEFAULT_OUTPUT_CSV="complete/dataset_all_with_images.csv"

# Drop rows with empty URL
DEFAULT_REQUIRE_URL=true

# ============================================
# Auto-configured paths
# ============================================

# Get project root (assuming script is in scripts/dataset_generation/ folder)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

RESULTS_DIR="${PROJECT_ROOT}/dataset_generation_output"
ITERATE_NAME="${DEFAULT_ITERATE_NAME}"
OUTPUT_CSV="${DEFAULT_OUTPUT_CSV}"
REQUIRE_URL="${DEFAULT_REQUIRE_URL}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --iterate-name <name>      Results subdir under ./dataset_generation_output (default: ${DEFAULT_ITERATE_NAME})
  --inputs <a> [b ...]       One or more JSON inputs (absolute or relative to repo root)
  --input-glob <glob>        Glob under dataset_generation_output/<iterate-name>/
  --output <path>            Output CSV (absolute or relative to dataset_generation_output/<iterate-name>/; default: ${DEFAULT_OUTPUT_CSV})
  --require-url              Drop rows with empty URL (default: ${DEFAULT_REQUIRE_URL})
  --no-require-url           Keep rows even if URL empty
  -h, --help                 Show help

Examples:
  $(basename "$0")
  $(basename "$0") --inputs dataset_generation_output/heldout/texts_by_merge_source_with_heldout_set/texts_hazard_augmented.json --output images/complete/hazard_augmented.csv --require-url
  $(basename "$0") --iterate-name heldout --input-glob "texts_by_merge_source_with_heldout_set/texts_*.json" --output images/complete/dataset.csv
EOF
}

CUSTOM_INPUTS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --iterate-name)
      ITERATE_NAME="$2"
      shift 2
      ;;
    --input-glob)
      INPUT_GLOB="$2"
      shift 2
      ;;
    --inputs)
      shift
      while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
        CUSTOM_INPUTS+=("$1")
        shift
      done
      ;;
    --output)
      OUTPUT_CSV="$2"
      shift 2
      ;;
    --require-url)
      REQUIRE_URL=true
      shift
      ;;
    --no-require-url)
      REQUIRE_URL=false
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      echo ""
      usage
      exit 2
      ;;
  esac
done

ITERATE_DIR="${RESULTS_DIR}/${ITERATE_NAME}"

if [[ "${OUTPUT_CSV}" = /* ]]; then
  OUT_PATH="${OUTPUT_CSV}"
else
  OUT_PATH="${ITERATE_DIR}/${OUTPUT_CSV}"
fi

INPUTS=()
if [ ${#CUSTOM_INPUTS[@]} -gt 0 ]; then
  # Use custom inputs
  INPUTS=( "${CUSTOM_INPUTS[@]}" )
elif [ -n "${INPUT_GLOB}" ]; then
  # Use glob pattern
  shopt -s nullglob
  GLOB_PATTERN="${ITERATE_DIR}/${INPUT_GLOB}"
  INPUTS=(${GLOB_PATTERN})
  shopt -u nullglob
else
  # Auto-detect texts_with_images_*.json files
  shopt -s nullglob
  INPUTS=("${ITERATE_DIR}"/texts_with_images_*.json)
  shopt -u nullglob
fi

if [ ${#INPUTS[@]} -eq 0 ]; then
  if [ ${#CUSTOM_INPUTS[@]} -gt 0 ]; then
    echo "Error: --inputs provided but no files found."
  elif [ -n "${INPUT_GLOB}" ]; then
    echo "Error: --input-glob '${INPUT_GLOB}' matched no files in ${ITERATE_DIR}"
  else
    echo "Error: no input files found in: ${ITERATE_DIR}"
    echo "Expected files matching pattern: texts_with_images_*.json"
  fi
  exit 1
fi

REQUIRE_URL_FLAG=""
if [ "${REQUIRE_URL}" = "true" ]; then
  REQUIRE_URL_FLAG="--require-url"
fi

echo "Make Dataset: ${ITERATE_NAME}"
echo "Found ${#INPUTS[@]} input file(s)"

# Split hazard_augmented files if needed
# If both hazard_augmented.json and hazard_augmented_split.json exist,
# only process the split version to avoid duplication
PROCESSED_INPUTS=()
HAS_SPLIT_FILE=false

# Check if split version already exists in INPUTS
for input_file in "${INPUTS[@]}"; do
  if [[ "$(basename "${input_file}")" == *"hazard_augmented"* ]] && [[ "$(basename "${input_file}")" == *"_split"* ]]; then
    HAS_SPLIT_FILE=true
    break
  fi
done

# Process files
for input_file in "${INPUTS[@]}"; do
  if [[ "$(basename "${input_file}")" == *"hazard_augmented"* ]] && [[ "$(basename "${input_file}")" != *"_split"* ]]; then
    # If split file already exists, skip the unsplit version
    if [ "${HAS_SPLIT_FILE}" = "true" ]; then
      echo "Skipping $(basename "${input_file}") - split version already exists"
      continue
    fi
    
    echo "Splitting dual scenarios: $(basename "${input_file}")"
    split_output="${input_file%.*}_split.json"
    
    python -c "
import sys
import os
sys.path.insert(0, '${PROJECT_ROOT}')
from src.dataset_generation.tasks.hazard_augmentation import HazardAugmentationTask
import argparse

# Create a dummy args object
args = argparse.Namespace()
task = HazardAugmentationTask(args)
task._split_all_dual_scenarios('${input_file}', '${split_output}', include_dual_metadata=True)
"
    
    PROCESSED_INPUTS+=("${split_output}")
    echo "  → Created: $(basename "${split_output}")"
  else
    PROCESSED_INPUTS+=("${input_file}")
  fi
done

echo ""
echo "Processing ${#PROCESSED_INPUTS[@]} file(s) to create dataset:"
for file in "${PROCESSED_INPUTS[@]}"; do
  echo "  - $(basename "${file}")"
done
echo "Output: ${OUT_PATH}"

mkdir -p "$(dirname "${OUT_PATH}")"

python ${PROJECT_ROOT}/src/dataset_generation/utils/json_to_dataset_csv.py \
  --inputs "${PROCESSED_INPUTS[@]}" \
  --output "${OUT_PATH}" \
  ${REQUIRE_URL_FLAG}

echo ""
echo "✓ Dataset created: ${OUT_PATH}"

# Generate statistics
echo ""
echo "Generating statistics..."

STATS_OUTPUT="${OUT_PATH%.*}_statistics.json"
python ${PROJECT_ROOT}/src/dataset_generation/utils/generate_dataset_statistics.py \
  --csv "${OUT_PATH}" \
  --output "${STATS_OUTPUT}"

echo ""
echo "✓ Statistics generated: ${STATS_OUTPUT}"
