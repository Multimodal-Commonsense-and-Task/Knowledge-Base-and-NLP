#!/bin/bash
# Step 3-3: Merge Graphs - New Task-based Version
# Usage: Edit configuration below and run: ./scripts/3_3_merge_graphs_new.sh

set -e  # Exit on any error

# ============================================
# Configuration - EDIT THESE
# ============================================

# Iteration name (results folder name)
ITERATE_NAME="train"

# Input files and labels (format: "filename:label")
# You can add as many as you want
INPUTS=(
  "graphs_normalized.json:original"
  "graphs_scene_augmented.json:scene_augmented"
  "graphs_normalized_hazard_removed.json:hazard_removed"
  "graphs_scene_augmented_hazard_removed.json:scene_aug_hazard_removed"
  "graphs_normalized_hazard_augmented.json:hazard_augmented"
)

# ============================================
# Auto-configured paths
# ============================================

# Get project root (assuming script is in scripts/dataset_generation/ folder)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

RESULTS_DIR="${PROJECT_ROOT}/dataset_generation_output"
ITERATE_DIR="${RESULTS_DIR}/${ITERATE_NAME}"
OUTPUT_PATH="${ITERATE_DIR}/graphs_final_for_image_generation.json"

# ============================================
# Pipeline Execution
# ============================================

echo "Merge Graphs Pipeline (New): ${ITERATE_NAME}"
mkdir -p "${ITERATE_DIR}"

# Check and collect available input files
echo "Checking available input files..."
AVAILABLE_FILES=()
AVAILABLE_LABELS=()

for input_spec in "${INPUTS[@]}"; do
    # Split by colon to get filename and label
    IFS=':' read -r filename label <<< "${input_spec}"
    file_path="${ITERATE_DIR}/${filename}"
    
    if [ -f "${file_path}" ]; then
        echo "  ✓ Found: ${filename} (label: ${label})"
        AVAILABLE_FILES+=("${file_path}")
        AVAILABLE_LABELS+=("${label}")
    else
        echo "  ✗ Not found: ${filename}"
    fi
done

echo "Total files to merge: ${#AVAILABLE_FILES[@]}"

if [ ${#AVAILABLE_FILES[@]} -eq 0 ]; then
    echo "Error: No input files found. Please check that at least one of the specified files exists."
    echo "Configured inputs:"
    for input_spec in "${INPUTS[@]}"; do
        IFS=':' read -r filename label <<< "${input_spec}"
        echo "  - ${filename} (${label})"
    done
    exit 1
fi

# Merge all available graph files
echo "Merging all available graph files..."
python ${PROJECT_ROOT}/src/dataset_generation/utils/merge_multiple_graph_files.py \
  --inputs "${AVAILABLE_FILES[@]}" \
  --labels "${AVAILABLE_LABELS[@]}" \
  --output "${OUTPUT_PATH}"

# Summary
echo ""
echo "✓ Merge Graphs completed: ${ITERATE_DIR}"
echo "  - graphs_final.json"
echo "  Total scenarios: $(python -c "import json; data=json.load(open('${OUTPUT_PATH}')); print(len(data.get('scenarios', data)) if isinstance(data, dict) else len(data))")"

