#!/bin/bash
# Restore image paths to texts_with_images_*.json files
# This script re-runs add_image_paths_to_scenarios for existing files
# Usage: ./scripts/5_3_restore_image_paths.sh

set -e  # Exit on any error

# Get project root (assuming script is in scripts/evaluation/ folder)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# ============================================
# Configuration - EDIT THESE
# ============================================

# Iteration name (results folder name)
ITERATE_NAME="train"

# Output directory where images are stored
OUTPUT_DIR="${PROJECT_ROOT}/dataset_generation_output/${ITERATE_NAME}/images/downloaded"

# Specific merge sources to restore (leave empty to restore all)
# Example: MERGE_SOURCES=("original" "hazard_removed")
MERGE_SOURCES=()

# ============================================
# Auto-configured paths
# ============================================

# Get project root (assuming script is in scripts/dataset_generation/ folder)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

RESULTS_DIR="${PROJECT_ROOT}/dataset_generation_output"
ITERATE_DIR="${RESULTS_DIR}/${ITERATE_NAME}"

if [ -z "${OUTPUT_DIR}" ] || [[ "${OUTPUT_DIR}" == *"\${PROJECT_ROOT}"* ]]; then
    OUTPUT_DIR="${ITERATE_DIR}/images/downloaded"
fi

# ============================================
# Pipeline Execution
# ============================================

echo "Restoring image paths to texts_with_images_*.json files: ${ITERATE_NAME}"
echo "Image directory: ${OUTPUT_DIR}"

if [ ! -d "${OUTPUT_DIR}" ]; then
    echo "Error: Image directory not found: ${OUTPUT_DIR}"
    exit 1
fi

# Determine which merge sources to process
if [ ${#MERGE_SOURCES[@]} -eq 0 ]; then
    # Find all texts_*.json files in texts_by_merge_source
    TEXT_FILES=("${ITERATE_DIR}/texts_by_merge_source"/texts_*.json)
    if [ ${#TEXT_FILES[@]} -eq 0 ] || [ ! -f "${TEXT_FILES[0]}" ]; then
        echo "Error: No text files found in ${ITERATE_DIR}/texts_by_merge_source/"
        exit 1
    fi
    
    # Extract merge sources from filenames
    for text_file in "${TEXT_FILES[@]}"; do
        MERGE_SOURCE=$(basename "${text_file}" | sed 's/texts_\(.*\)\.json/\1/')
        MERGE_SOURCES+=("${MERGE_SOURCE}")
    done
fi

echo "Processing ${#MERGE_SOURCES[@]} merge source(s):"
for ms in "${MERGE_SOURCES[@]}"; do
    echo "  - ${ms}"
done
echo ""

# Process each merge source
RESTORED_COUNT=0
for MERGE_SOURCE in "${MERGE_SOURCES[@]}"; do
    TEXT_FILE="${ITERATE_DIR}/texts_by_merge_source/texts_${MERGE_SOURCE}.json"
    OUTPUT_WITH_IMAGES="${ITERATE_DIR}/texts_with_images_${MERGE_SOURCE}.json"
    
    if [ ! -f "${TEXT_FILE}" ]; then
        echo "Warning: Text file not found: ${TEXT_FILE}"
        continue
    fi
    
    echo "Processing merge source: ${MERGE_SOURCE}"
    echo "  Input: $(basename ${TEXT_FILE})"
    echo "  Output: $(basename ${OUTPUT_WITH_IMAGES})"
    
    # Count scenarios before
    SCENARIOS_BEFORE=$(python -c "
import json
with open('${TEXT_FILE}', 'r') as f:
    data = json.load(f)
scenarios = data.get('scenarios', data)
print(len(scenarios))
" 2>/dev/null)
    
    # Add image paths
    python -c "
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
from src.dataset_generation.utils.img_utils import add_image_paths_to_scenarios
add_image_paths_to_scenarios(
    '${TEXT_FILE}',
    '${OUTPUT_DIR}',
    '${OUTPUT_WITH_IMAGES}',
    '${MERGE_SOURCE}'
)
"
    
    # Count scenarios with images after
    SCENARIOS_WITH_IMAGES=$(python -c "
import json
with open('${OUTPUT_WITH_IMAGES}', 'r') as f:
    data = json.load(f)
scenarios = data.get('scenarios', data)
with_images = sum(1 for s in scenarios if s.get('image_path') or s.get('url'))
print(with_images)
" 2>/dev/null)
    
    echo "  ✓ Restored: ${SCENARIOS_WITH_IMAGES}/${SCENARIOS_BEFORE} scenarios with images"
    echo ""
    
    if [ "${SCENARIOS_WITH_IMAGES}" -gt 0 ]; then
        RESTORED_COUNT=$((RESTORED_COUNT + 1))
    fi
done

# Summary
echo "✓ Image path restoration completed: ${ITERATE_DIR}"
echo "  - Restored ${RESTORED_COUNT}/${#MERGE_SOURCES[@]} merge source(s)"
echo "  - Image directory: ${OUTPUT_DIR}"
