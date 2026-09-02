#!/bin/bash

# EMBGuard Results Aggregation Script
# Aggregates evaluation results across multiple runs and generates statistics
# Usage: bash scripts/evaluation/aggregate_results.sh

# Get project root (assuming script is in scripts/evaluation/ folder)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# ============================================
# Configuration
# ============================================
RESULTS_DIR="results/EMBGuardTest"  # Directory containing Run_1, Run_2, ... folders
OUTPUT_FILE="results/EMBGuardTest/aggregated_results.txt"  # Output file for aggregated results
RUN_PREFIX="Run_"  # Prefix for run directories (default: Run_)
# ============================================

echo "============================================================"
echo "EMBGuard Results Aggregation"
echo "============================================================"
echo "Results directory: $RESULTS_DIR"
echo "Output file: $OUTPUT_FILE"
echo "Run prefix: $RUN_PREFIX"
echo "============================================================"
echo ""

# Run aggregation script
python src/evals/aggregate_results.py \
    --results-dir "$RESULTS_DIR" \
    --output-file "$OUTPUT_FILE" \
    --run-prefix "$RUN_PREFIX"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✓ Aggregation completed successfully!"
    echo "============================================================"
    echo ""
    echo "Generated files:"
    echo "  - Aggregated results: $OUTPUT_FILE"
    echo "  - Individual statistics: $RESULTS_DIR/Run_*/model_name/single_inference_statistics.txt"
    echo ""
else
    echo ""
    echo "============================================================"
    echo "✗ Aggregation failed (exit code: $exit_code)"
    echo "============================================================"
    exit $exit_code
fi
