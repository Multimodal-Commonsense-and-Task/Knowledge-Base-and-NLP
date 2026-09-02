#!/bin/bash

# EMBGuard Correlation Visualization Script
# Usage: bash scripts/visualization/plot_correlation.sh
# Creates a combined visualization of correlation results for all three metrics

# Get project root (assuming script is in scripts/visualization/ folder)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# ============================================
# Configuration - Edit these values as needed
# ============================================
# Correlation CSV files (from run_calculate_correlation.sh output)
CORRELATION_DIR="results/correlation"
CONDITION="non-thinking"  # Match the condition used in run_calculate_correlation.sh
CORRELATION_METHOD="pearson"  # or "spearman"

# Input CSV files
POTENTIAL_RISK_CSV="${CORRELATION_DIR}/correlation_all_metrics_${CONDITION}_potential-risk_${CORRELATION_METHOD}.csv"
CONDITIONAL_RISK_TYPE_CSV="${CORRELATION_DIR}/correlation_all_metrics_${CONDITION}_conditional-risk-type_${CORRELATION_METHOD}.csv"
CONDITIONAL_HAZARD_CSV="${CORRELATION_DIR}/correlation_all_metrics_${CONDITION}_conditional-hazard_${CORRELATION_METHOD}.csv"

# Output file
OUTPUT_DIR="results/figures"
OUTPUT_FILE="${OUTPUT_DIR}/correlation_combined_${CONDITION}_${CORRELATION_METHOD}.png"

# Optional title
TITLE=""
# ============================================

echo "============================================================"
echo "EMBGuard Correlation Visualization"
echo "============================================================"
echo "Potential Risk CSV: $POTENTIAL_RISK_CSV"
echo "Conditional Risk Type CSV: $CONDITIONAL_RISK_TYPE_CSV"
echo "Conditional Hazard CSV: $CONDITIONAL_HAZARD_CSV"
echo "Output file: $OUTPUT_FILE"
echo "Correlation method: $CORRELATION_METHOD"
echo "============================================================"
echo ""

# Check if input files exist
if [ ! -f "$POTENTIAL_RISK_CSV" ]; then
    echo "Error: Potential Risk CSV file not found: $POTENTIAL_RISK_CSV"
    exit 1
fi

if [ ! -f "$CONDITIONAL_RISK_TYPE_CSV" ]; then
    echo "Error: Conditional Risk Type CSV file not found: $CONDITIONAL_RISK_TYPE_CSV"
    exit 1
fi

if [ ! -f "$CONDITIONAL_HAZARD_CSV" ]; then
    echo "Error: Conditional Hazard CSV file not found: $CONDITIONAL_HAZARD_CSV"
    exit 1
fi

# Build command
CMD="python src/visualization/plot_correlation.py"
CMD="$CMD --potential-risk-csv $POTENTIAL_RISK_CSV"
CMD="$CMD --conditional-risk-type-csv $CONDITIONAL_RISK_TYPE_CSV"
CMD="$CMD --conditional-hazard-csv $CONDITIONAL_HAZARD_CSV"
CMD="$CMD --output-file $OUTPUT_FILE"
CMD="$CMD --correlation-method $CORRELATION_METHOD"

if [ -n "$TITLE" ]; then
    CMD="$CMD --title \"$TITLE\""
fi

# Run visualization
echo "Generating correlation plot..."
echo ""
$CMD

echo ""
echo "============================================================"
echo "Visualization completed!"
echo "============================================================"
echo "Plot saved to: $OUTPUT_FILE"
echo "============================================================"
