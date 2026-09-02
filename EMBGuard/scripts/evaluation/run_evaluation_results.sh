#!/bin/bash

# EMBGuard Results Evaluation Script
# Evaluates inference results using a judge model (LLM-as-a-Judge)
# Usage: bash scripts/evaluation/run_evaluation_results.sh
#
# This script evaluates JSONL result files in outputs/ directories
# and saves evaluation results to results/ directories

# Get project root (assuming script is in scripts/evaluation/ folder)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# ============================================
# Hugging Face Token Configuration
# ============================================
# Set HF_TOKEN for downloading datasets from Hugging Face Hub
# Option 1: Set environment variable before running script: export HF_TOKEN="your_token"
# Option 2: Set it here directly (uncomment and edit):
# HF_TOKEN="your_huggingface_token_here"

if [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN
    echo "HF_TOKEN is set"
fi

# ============================================
# Configuration - Edit these arrays as needed
# ============================================
# Provider-Model pairs to evaluate results for
# Format: "provider:model" or "vllm:model:port" (port only for vLLM)

PROVIDER_MODEL_PAIRS=(
    "openai:gpt-4o-mini"
    "openai:gpt-4o"
    "openai:gpt-5.1"
    # "openrouter:qwen/qwen3-vl-8b-instruct"
    # "vllm:Qwen/Qwen3-VL-4B-Instruct:8000"
    # "gemini:gemini-2.5-flash"
    # "gemini:gemini-2.5-pro"
)

# Filter settings - only evaluate files matching these settings
USE_FEW_SHOT="false"   # Set to "true" to evaluate few-shot results, "false" for no-few-shot
USE_THINKING="false"   # Set to "true" to evaluate thinking results, "false" for non-thinking

# Evaluation type selection
EVAL_TEST_SET="true"     # Set to "true" to evaluate test set results
EVAL_HELDOUT_SET="false" # Set to "true" to evaluate heldout set results

# Judge model configuration
JUDGE_PROVIDER="openai"
JUDGE_MODEL="gpt-4o"
NUM_WORKERS="16"
# ============================================

# Function to evaluate test set results
evaluate_test_set_results() {
    local provider="$1"
    local model="$2"
    
    # Clean model name for path (replace / with _)
    local model_clean=$(echo "$model" | sed 's/\//_/g')
    local provider_model="${provider}_${model_clean}"
    local results_dir="outputs/EMBGuardTest/${provider_model}"
    
    if [ ! -d "$results_dir" ]; then
        echo "Warning: Results directory not found: $results_dir"
        return 1
    fi
    
    echo ""
    echo "============================================================"
    echo "Evaluating Test Set Results: $provider / $model"
    echo "============================================================"
    echo ""
    
    # Determine file pattern based on settings
    local few_shot_suffix=""
    local thinking_suffix=""
    
    if [ "$USE_FEW_SHOT" = "false" ]; then
        few_shot_suffix="no-few-shot"
    else
        few_shot_suffix="few-shot"
    fi
    
    if [ "$USE_THINKING" = "true" ]; then
        thinking_suffix="thinking"
    else
        thinking_suffix="non-thinking"
    fi
    
    local found_files=0
    for jsonl_file in "$results_dir"/*.jsonl; do
        if [ -f "$jsonl_file" ]; then
            local basename=$(basename "$jsonl_file" .jsonl)
            
            # Skip files that don't match current settings
            if [[ ! "$basename" =~ "${few_shot_suffix}" ]] || [[ ! "$basename" =~ "${thinking_suffix}" ]]; then
                continue
            fi
            
            local eval_results_dir="results/EMBGuardTest/${provider_model}"
            mkdir -p "$eval_results_dir"
            local output_file="${eval_results_dir}/${basename}_evaluation.json"
            
            echo "Evaluating: $jsonl_file"
            echo "  Output: $output_file"
            python src/evaluate.py results \
                --results-file "$jsonl_file" \
                --output-file "$output_file" \
                --judge-provider "$JUDGE_PROVIDER" \
                --judge-model "$JUDGE_MODEL" \
                --num-workers "$NUM_WORKERS"
            echo ""
            found_files=$((found_files + 1))
        fi
    done
    
    if [ $found_files -eq 0 ]; then
        echo "No JSONL files found in $results_dir matching settings (${few_shot_suffix}, ${thinking_suffix})"
        return 1
    fi
    
    echo "✓ Evaluated $found_files test set result file(s)"
    return 0
}

# Function to evaluate heldout set results
evaluate_heldout_set_results() {
    local provider="$1"
    local model="$2"
    
    # Clean model name for path (replace / with _)
    local model_clean=$(echo "$model" | sed 's/\//_/g')
    local provider_model="${provider}_${model_clean}"
    local results_dir="outputs/heldout_set/${provider_model}"
    
    if [ ! -d "$results_dir" ]; then
        echo "Warning: Results directory not found: $results_dir"
        return 1
    fi
    
    echo ""
    echo "============================================================"
    echo "Evaluating Heldout Set Results: $provider / $model"
    echo "============================================================"
    echo ""
    
    # Determine file pattern based on settings
    local few_shot_suffix=""
    local thinking_suffix=""
    
    if [ "$USE_FEW_SHOT" = "false" ]; then
        few_shot_suffix="no-few-shot"
    else
        few_shot_suffix="few-shot"
    fi
    
    if [ "$USE_THINKING" = "true" ]; then
        thinking_suffix="thinking"
    else
        thinking_suffix="non-thinking"
    fi
    
    local found_files=0
    for jsonl_file in "$results_dir"/*.jsonl; do
        if [ -f "$jsonl_file" ]; then
            local basename=$(basename "$jsonl_file" .jsonl)
            
            # Skip files that don't match current settings
            if [[ ! "$basename" =~ "${few_shot_suffix}" ]] || [[ ! "$basename" =~ "${thinking_suffix}" ]]; then
                continue
            fi
            
            local eval_results_dir="results/heldout_set/${provider_model}"
            mkdir -p "$eval_results_dir"
            local output_file="${eval_results_dir}/${basename}_evaluation.json"
            
            echo "Evaluating: $jsonl_file"
            echo "  Output: $output_file"
            python src/evals/evaluate_heldout_results.py \
                --results-file "$jsonl_file" \
                --output-file "$output_file" \
                --judge-provider "$JUDGE_PROVIDER" \
                --judge-model "$JUDGE_MODEL" \
                --num-workers "$NUM_WORKERS"
            echo ""
            found_files=$((found_files + 1))
        fi
    done
    
    if [ $found_files -eq 0 ]; then
        echo "No JSONL files found in $results_dir matching settings (${few_shot_suffix}, ${thinking_suffix})"
        return 1
    fi
    
    echo "✓ Evaluated $found_files heldout set result file(s)"
    return 0
}

# ============================================
# Main execution
# ============================================
total_pairs=${#PROVIDER_MODEL_PAIRS[@]}
current=0

echo "============================================================"
echo "EMBGuard Results Evaluation"
echo "============================================================"
echo "Total model pairs: $total_pairs"
echo "Judge: $JUDGE_PROVIDER / $JUDGE_MODEL"
echo "Workers: $NUM_WORKERS"
echo "Filter - Few-shot: $USE_FEW_SHOT, Thinking: $USE_THINKING"
echo "Evaluate Test Set: $EVAL_TEST_SET"
echo "Evaluate Heldout Set: $EVAL_HELDOUT_SET"
echo "============================================================"
echo ""

for pair in "${PROVIDER_MODEL_PAIRS[@]}"; do
    # Skip commented lines
    if [[ "$pair" =~ ^[[:space:]]*# ]]; then
        continue
    fi
    
    # Split provider:model:port pair (port is optional, only for vLLM)
    IFS=':' read -r provider model port <<< "$pair"
    
    # Trim whitespace
    provider=$(echo "$provider" | xargs)
    model=$(echo "$model" | xargs)
    port=$(echo "$port" | xargs)
    
    current=$((current + 1))
    echo "[$current/$total_pairs] Processing: $provider / $model"
    
    # Evaluate test set results
    if [ "$EVAL_TEST_SET" = "true" ]; then
        evaluate_test_set_results "$provider" "$model"
    fi
    
    # Evaluate heldout set results
    if [ "$EVAL_HELDOUT_SET" = "true" ]; then
        evaluate_heldout_set_results "$provider" "$model"
    fi
    
    echo ""
done

echo "============================================================"
echo "All evaluations completed!"
echo "============================================================"
