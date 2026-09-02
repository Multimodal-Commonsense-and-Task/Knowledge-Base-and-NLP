#!/bin/bash

# EMBGuard Scenario Type Trend Visualization Script
# Usage: bash scripts/visualization/plot_type_trend.sh
# Creates a line plot showing potential_risk trends across EMBGuardTest scenario types:
# Causal Risky, Selective Risky, Decoupled Benign, and Absent Benign.

# Get project root (assuming script is in scripts/visualization/ folder)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# ============================================
# Configuration - Edit these values as needed
# ============================================
# Input CSV file (scenario_type-based results)
CSV_FILE="results/EMBGuardTest/aggregated_results_by_scenario_type.csv"

# Output file
OUTPUT_DIR="results/figures"
OUTPUT_FILE="${OUTPUT_DIR}/scenario_type_trend_potential_risk.png"

# Optional: Specify models to plot (leave empty to plot all models)
# Example: MODELS=("openai_gpt-4o" "gemini_gemini-2.5-pro" "vllm_EMBGuard_EMBGuard-4B")
MODELS=()

# Optional title
TITLE=""
# ============================================

echo "============================================================"
echo "EMBGuard Scenario Type Trend Visualization"
echo "============================================================"
echo "CSV file: $CSV_FILE"
echo "Output file: $OUTPUT_FILE"
if [ ${#MODELS[@]} -gt 0 ]; then
    echo "Models: ${MODELS[*]}"
else
    echo "Models: (all models)"
fi
echo "============================================================"
echo ""

# Check if input file exists
if [ ! -f "$CSV_FILE" ]; then
    echo "Error: CSV file not found: $CSV_FILE"
    exit 1
fi

# Build command
CMD="python src/visualization/plot_type_trend.py"
CMD="$CMD --csv-file $CSV_FILE"
CMD="$CMD --output-file $OUTPUT_FILE"

if [ ${#MODELS[@]} -gt 0 ]; then
    CMD="$CMD --models ${MODELS[*]}"
fi

if [ -n "$TITLE" ]; then
    CMD="$CMD --title \"$TITLE\""
fi

# Run visualization
echo "Generating type trend plot..."
echo ""
$CMD

echo ""
echo "============================================================"
echo "Visualization completed!"
echo "============================================================"
echo "Plot saved to: $OUTPUT_FILE"
echo "============================================================"
