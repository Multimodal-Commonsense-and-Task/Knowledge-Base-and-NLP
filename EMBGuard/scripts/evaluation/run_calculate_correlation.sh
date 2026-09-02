#!/bin/bash

# EMBGuard Correlation Analysis Script
# Usage: bash scripts/run_calculate_correlation.sh
# Calculates correlation between test set and heldout set scores

# Get project root (assuming script is in scripts/evaluation/ folder)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# ============================================
# Configuration - Edit these values as needed
# ============================================
# Metric to use for correlation
# Leave empty to calculate for all three: potential_risk, conditional_risk_type, conditional_hazard
# Options: overall_accuracy, potential_risk_accuracy, risk_type_accuracy, hazard_accuracy,
#          conditional_risk_type_accuracy, conditional_hazard_accuracy
METRIC=""

# Condition filter (optional)
# Set to "non-thinking", "thinking", or leave empty for all conditions
CONDITION="non-thinking"

# Heldout dataset type filter (optional)
# Set to "safe", "unsafe", or leave empty for combined
HELDOUT_DATASET_TYPE=""

# Correlation method
CORRELATION_METHOD="pearson"
# Options: pearson, spearman

# CSV file paths (optional - if specified, uses CSV instead of scanning results directory)
TEST_CSV="results/EMBGuardTest/aggregated_results_overall_percentage.csv"
HELDOUT_CSV="results/heldout_set/aggregated_results_overall_percentage.csv"
# Leave empty to use results directory scanning instead
# TEST_CSV=""
# HELDOUT_CSV=""

# Output settings
OUTPUT_DIR="results/correlation"
if [ -n "$METRIC" ]; then
    OUTPUT_FILE="${OUTPUT_DIR}/correlation_${METRIC}"
else
    OUTPUT_FILE="${OUTPUT_DIR}/correlation_all_metrics"
fi
if [ -n "$CONDITION" ]; then
    OUTPUT_FILE="${OUTPUT_FILE}_${CONDITION}"
fi
if [ -n "$HELDOUT_DATASET_TYPE" ]; then
    OUTPUT_FILE="${OUTPUT_FILE}_${HELDOUT_DATASET_TYPE}"
fi
OUTPUT_FILE="${OUTPUT_FILE}.csv"

# Generate plot
GENERATE_PLOT=true
# ============================================

echo "============================================================"
echo "EMBGuard Correlation Analysis"
echo "============================================================"
if [ -n "$METRIC" ]; then
    echo "Metric: $METRIC"
else
    echo "Metric: (all three: potential_risk, conditional_risk_type, conditional_hazard)"
fi
if [ -n "$TEST_CSV" ]; then
    echo "Test set CSV: $TEST_CSV"
else
    echo "Test set results directory: results/EMBGuardTest"
fi
if [ -n "$HELDOUT_CSV" ]; then
    echo "Heldout set CSV: $HELDOUT_CSV"
else
    echo "Heldout set results directory: results/heldout_set"
fi
if [ -n "$CONDITION" ]; then
    echo "Condition filter: $CONDITION"
else
    echo "Condition filter: (all conditions)"
fi
if [ -n "$HELDOUT_DATASET_TYPE" ]; then
    echo "Heldout dataset type: $HELDOUT_DATASET_TYPE"
else
    echo "Heldout dataset type: (combined)"
fi
echo "Correlation method: $CORRELATION_METHOD"
echo "Output file: $OUTPUT_FILE"
echo "Generate plot: $GENERATE_PLOT"
echo "============================================================"
echo ""

# Build command
CMD="python src/evals/calculate_correlation.py"
if [ -n "$METRIC" ]; then
    CMD="$CMD --metric $METRIC"
fi
if [ -n "$TEST_CSV" ]; then
    CMD="$CMD --test-csv $TEST_CSV"
fi
if [ -n "$HELDOUT_CSV" ]; then
    CMD="$CMD --heldout-csv $HELDOUT_CSV"
fi
CMD="$CMD --correlation-method $CORRELATION_METHOD"
CMD="$CMD --output-file $OUTPUT_FILE"

if [ -n "$CONDITION" ]; then
    CMD="$CMD --condition $CONDITION"
fi

if [ -n "$HELDOUT_DATASET_TYPE" ]; then
    CMD="$CMD --heldout-dataset-type $HELDOUT_DATASET_TYPE"
fi

if [ "$GENERATE_PLOT" = true ]; then
    CMD="$CMD --plot"
    # Plot file will be auto-generated with metric suffix if multiple metrics
    # So we don't need to specify --plot-file
fi

# Run correlation analysis
echo "Running correlation analysis..."
echo ""
$CMD

echo ""
echo "============================================================"
echo "Correlation analysis completed!"
echo "============================================================"
if [ -n "$METRIC" ]; then
    echo "Results saved to: $OUTPUT_FILE"
    if [ "$GENERATE_PLOT" = true ]; then
        PLOT_FILE="${OUTPUT_FILE%.csv}.png"
        echo "Plot saved to: $PLOT_FILE"
    fi
else
    echo "Results saved to: ${OUTPUT_FILE%.csv}_*.csv (one file per metric)"
    if [ "$GENERATE_PLOT" = true ]; then
        echo "Plots saved to: ${OUTPUT_FILE%.csv}_*.png (one plot per metric)"
    fi
fi
echo "Summary saved to: ${OUTPUT_FILE%.csv}.summary.txt"
echo "============================================================"

