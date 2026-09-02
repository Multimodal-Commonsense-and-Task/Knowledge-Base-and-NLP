#!/bin/bash

# EMBGuard Unified Evaluation Script
# Runs both EMBGuardTest and heldout_set evaluations, and automatically evaluates results
# Usage: bash scripts/evaluation/run_all_evaluations.sh [vllm_port]
#   - vllm_port: Optional port number for vLLM (e.g., 8000, 8008)
#     If not provided, uses VLLM_PORT variable or default from config.yaml
# 
# To evaluate only results (skip inference), set:
#   RUN_TEST_SET="false"
#   RUN_HELDOUT_SET="false"
#   RUN_RESULTS_EVAL="true"
#   Results will be evaluated for all models in PROVIDER_MODEL_PAIRS
# 
# Edit PROVIDER_MODEL_PAIRS array below to run multiple combinations

# Get project root (assuming script is in scripts/evaluation/ folder)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# ============================================
# Hugging Face Token Configuration
# ============================================
# Set HF_TOKEN for downloading datasets from Hugging Face Hub
# Option 1: Set environment variable before running script: export HF_TOKEN="your_token"
# Option 2: Set it here directly (uncomment and edit):
HF_TOKEN="xxxx"

# If HF_TOKEN is set, export it so Python scripts can access it
if [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN
    echo "HF_TOKEN is set"
else
    echo "Warning: HF_TOKEN is not set. You may not be able to download private datasets from Hugging Face Hub."
    echo "Set it with: export HF_TOKEN='your_token' or edit this script directly."
fi

# ============================================
# Configuration - Edit these arrays as needed
# ============================================
# Provider-Model pairs to evaluate
# Format: "provider:model" or "vllm:model:port" (port only for vLLM)
# Examples:
#   "openai:gpt-4o"
#   "vllm:Qwen/Qwen3-VL-2B-Instruct:8000"
#   "vllm:Qwen/Qwen2-VL-2B-Instruct:8008"


PROVIDER_MODEL_PAIRS=(
    "openai:gpt-4o-mini"
    "openai:gpt-4o"
    "openai:gpt-5.1"
    "openrouter:qwen/qwen3-vl-8b-instruct"
    "openrouter:qwen/qwen3-vl-32b-instruct"
    "openrouter:qwen/qwen3-vl-30b-a3b-instruct"
    "openrouter:qwen/qwen3-vl-235b-a22b-instruct"
    # "vllm:OpenGVLab/InternVL3_5-1B-HF:8000"
    # "vllm:OpenGVLab/InternVL3_5-2B-HF:8008"
    "openrouter:google/gemma-3-4b-it"
    "openrouter:google/gemma-3-12b-it"
    "openrouter:google/gemma-3-27b-it"
    "openrouter:meta-llama/llama-3.2-11b-vision-instruct"
    "openrouter:meta-llama/llama-3.2-90b-vision-instruct"
    "vllm:Qwen/Qwen3-VL-4B-Instruct:8003"
    "vllm:Qwen/Qwen3-VL-2B-Instruct:8002"
    # "openrouter:opengvlab/internvl3-14b"
    # "openrouter:opengvlab/internvl3-78b"
    "openrouter:google/gemini-2.5-flash"
    "openrouter:google/gemini-2.5-pro"
    # "gemini:gemini-2.5-flash"
    # "gemini:gemini-2.5-pro"
    "vllm:EMBGuard/EMBGuard-2B:8000"
    "vllm:EMBGuard/EMBGuard-4B:8001"
)

# Test set configuration
TEST_SET="all"  # Options: "all", "causal_risky", "selective_risky", "decoupled_benign", "absent_benign", or comma-separated
TEST_SET_NUM_WORKERS="32"

# Heldout set configuration
HELDOUT_DATASET="all"  # Options: "all", "safe", "unsafe", or comma-separated (e.g., "safe,unsafe")
HELDOUT_NUM_WORKERS="32"

# Common configuration
USE_FEW_SHOT="false"  # Set to "true" to enable few-shot examples
USE_THINKING="true"   # Set to "true" to enable thinking mode (step-by-step reasoning)

# Evaluation mode selection
RUN_TEST_SET="true"      # Set to "false" to skip test set evaluation
RUN_HELDOUT_SET="false"   # Set to "false" to skip heldout set evaluation
RUN_RESULTS_EVAL="true"  # Set to "false" to skip results evaluation (runs after inference)

# Results evaluation configuration (runs automatically after inference)
JUDGE_PROVIDER="openai"
JUDGE_MODEL="gpt-4o"
RESULTS_NUM_WORKERS="16"

# Parallel execution configuration
PARALLEL_MODE="true"     # Set to "true" to run all models in parallel
MAX_PARALLEL_JOBS=8       # Maximum number of parallel jobs (0 = unlimited)

# Multiple runs configuration
NUM_RUNS=1                # Number of times to run each evaluation (creates Run_1, Run_2, ... folders)
# ============================================

# Function to run test set evaluation for a single provider-model combination
run_test_set_evaluation() {
    local provider="$1"
    local model="$2"
    local port="$3"  # Optional port for vLLM
    local run_num="$4"  # Run number (for Run_n folder structure)
    
    echo ""
    echo "============================================================"
    echo "Running Test Set Evaluation: $provider / $model"
    [ -n "$port" ] && echo "Port: $port"
    [ -n "$run_num" ] && echo "Run: $run_num"
    echo "============================================================"
    echo ""
    
    # Pass vLLM port if specified (will override port in config.yaml base_url)
    local vllm_port_arg=""
    if [ "$provider" = "vllm" ] && [ -n "$port" ]; then
        vllm_port_arg="--vllm-port $port"
        echo "Using vLLM port: $port (overriding config.yaml)"
    fi
    
    # Build output directory: DATASET_NAME/Run_n/Model_name
    local model_clean=$(echo "$model" | sed 's/\//_/g')
    local output_dir_arg=""
    if [ -n "$run_num" ]; then
        local output_dir="outputs/EMBGuardTest/Run_${run_num}/${provider}_${model_clean}"
        output_dir_arg="--output_dir $output_dir"
        echo "Output directory: $output_dir"
    fi
    
    python src/evaluate.py test-set \
        --provider "$provider" \
        --model "$model" \
        --test-set "$TEST_SET" \
        --num-workers "$TEST_SET_NUM_WORKERS" \
        $([ "$USE_FEW_SHOT" = "false" ] && echo "--no-few-shot" || echo "") \
        $([ "$USE_THINKING" = "true" ] && echo "--use-thinking" || echo "") \
        $vllm_port_arg \
        $output_dir_arg
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "✓ Test Set Completed: $provider / $model"
    else
        echo ""
        echo "✗ Test Set Failed: $provider / $model (exit code: $exit_code)"
    fi
    
    return $exit_code
}

# Function to run heldout set evaluation for a single provider-model combination
run_heldout_set_evaluation() {
    local provider="$1"
    local model="$2"
    local port="$3"  # Optional port for vLLM
    local run_num="$4"  # Run number (for Run_n folder structure)
    
    echo ""
    echo "============================================================"
    echo "Running Heldout Set Evaluation: $provider / $model"
    [ -n "$port" ] && echo "Port: $port"
    [ -n "$run_num" ] && echo "Run: $run_num"
    echo "============================================================"
    echo ""
    
    # Pass vLLM port if specified (will override port in config.yaml base_url)
    local vllm_port_arg=""
    if [ "$provider" = "vllm" ] && [ -n "$port" ]; then
        vllm_port_arg="--vllm-port $port"
        echo "Using vLLM port: $port (overriding config.yaml)"
    fi
    
    # Build output directory: DATASET_NAME/Run_n/Model_name
    local model_clean=$(echo "$model" | sed 's/\//_/g')
    local output_dir_arg=""
    if [ -n "$run_num" ]; then
        local output_dir="outputs/heldout_set/Run_${run_num}/${provider}_${model_clean}"
        output_dir_arg="--output_dir $output_dir"
        echo "Output directory: $output_dir"
    fi
    
    python src/evaluate.py heldout-set \
        --provider "$provider" \
        --model "$model" \
        --dataset "$HELDOUT_DATASET" \
        --num-workers "$HELDOUT_NUM_WORKERS" \
        $([ "$USE_FEW_SHOT" = "false" ] && echo "--no-few-shot" || echo "") \
        $([ "$USE_THINKING" = "true" ] && echo "--use-thinking" || echo "") \
        $vllm_port_arg \
        $output_dir_arg
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "✓ Heldout Set Completed: $provider / $model"
    else
        echo ""
        echo "✗ Heldout Set Failed: $provider / $model (exit code: $exit_code)"
    fi
    
    return $exit_code
}

# Function to evaluate test set results
evaluate_test_set_results() {
    local provider="$1"
    local model="$2"
    local run_num="$3"  # Run number (for Run_n folder structure)
    
    # Clean model name for path (replace / with _)
    local model_clean=$(echo "$model" | sed 's/\//_/g')
    local provider_model="${provider}_${model_clean}"
    
    # Build results directory path
    local results_dir=""
    if [ -n "$run_num" ]; then
        results_dir="outputs/EMBGuardTest/Run_${run_num}/${provider_model}"
    else
        results_dir="outputs/EMBGuardTest/${provider_model}"
    fi
    
    if [ ! -d "$results_dir" ]; then
        echo "Warning: Results directory not found: $results_dir"
        return 1
    fi
    
    echo ""
    echo "============================================================"
    echo "Evaluating Test Set Results: $provider / $model"
    echo "============================================================"
    echo ""
    
    # Find JSONL files matching current run settings
    # Only evaluate files that match the current few-shot and thinking settings
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
            
            # Skip files that don't match current settings (few-shot and thinking)
            if [[ ! "$basename" =~ "${few_shot_suffix}" ]] || [[ ! "$basename" =~ "${thinking_suffix}" ]]; then
                continue
            fi
            
            # Build evaluation results directory path
            local eval_results_dir=""
            if [ -n "$run_num" ]; then
                eval_results_dir="results/EMBGuardTest/Run_${run_num}/${provider_model}"
            else
                eval_results_dir="results/EMBGuardTest/${provider_model}"
            fi
            mkdir -p "$eval_results_dir"
            local output_file="${eval_results_dir}/${basename}_evaluation.json"
            
            echo "Evaluating: $jsonl_file"
            python src/evaluate.py results \
                --results-file "$jsonl_file" \
                --output-file "$output_file" \
                --judge-provider "$JUDGE_PROVIDER" \
                --judge-model "$JUDGE_MODEL" \
                --num-workers "$RESULTS_NUM_WORKERS"
            echo ""
            found_files=$((found_files + 1))
        fi
    done
    
    if [ $found_files -eq 0 ]; then
        echo "No JSONL files found in $results_dir matching current settings (${few_shot_suffix}, ${thinking_suffix})"
        return 1
    fi
    
    echo "✓ Evaluated $found_files test set result file(s)"
    return 0
}

# Function to evaluate heldout set results
evaluate_heldout_set_results() {
    local provider="$1"
    local model="$2"
    local run_num="$3"  # Run number (for Run_n folder structure)
    
    # Clean model name for path (replace / with _)
    local model_clean=$(echo "$model" | sed 's/\//_/g')
    local provider_model="${provider}_${model_clean}"
    
    # Build results directory path
    local results_dir=""
    if [ -n "$run_num" ]; then
        results_dir="outputs/heldout_set/Run_${run_num}/${provider_model}"
    else
        results_dir="outputs/heldout_set/${provider_model}"
    fi
    
    if [ ! -d "$results_dir" ]; then
        echo "Warning: Results directory not found: $results_dir"
        return 1
    fi
    
    echo ""
    echo "============================================================"
    echo "Evaluating Heldout Set Results: $provider / $model"
    echo "============================================================"
    echo ""
    
    # Find JSONL files matching current run settings
    # Only evaluate files that match the current few-shot and thinking settings
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
            
            # Skip files that don't match current settings (few-shot and thinking)
            if [[ ! "$basename" =~ "${few_shot_suffix}" ]] || [[ ! "$basename" =~ "${thinking_suffix}" ]]; then
                continue
            fi
            
            # Build evaluation results directory path
            local eval_results_dir=""
            if [ -n "$run_num" ]; then
                eval_results_dir="results/heldout_set/Run_${run_num}/${provider_model}"
            else
                eval_results_dir="results/heldout_set/${provider_model}"
            fi
            mkdir -p "$eval_results_dir"
            local output_file="${eval_results_dir}/${basename}_evaluation.json"
            
            echo "Evaluating: $jsonl_file"
            python src/evals/evaluate_heldout_results.py \
                --results-file "$jsonl_file" \
                --output-file "$output_file" \
                --judge-provider "$JUDGE_PROVIDER" \
                --judge-model "$JUDGE_MODEL" \
                --num-workers "$RESULTS_NUM_WORKERS"
            echo ""
            found_files=$((found_files + 1))
        fi
    done
    
    if [ $found_files -eq 0 ]; then
        echo "No JSONL files found in $results_dir matching current settings (${few_shot_suffix}, ${thinking_suffix})"
        return 1
    fi
    
    echo "✓ Evaluated $found_files heldout set result file(s)"
    return 0
}

# Function to run all evaluations for a single provider-model combination
run_all_evaluations() {
    local provider="$1"
    local model="$2"
    local port="$3"  # Optional port for vLLM
    local run_num="$4"  # Run number (for Run_n folder structure)
    
    local test_exit_code=0
    local heldout_exit_code=0
    local test_results_exit_code=0
    local heldout_results_exit_code=0
    
    # Run test set evaluation if enabled
    if [ "$RUN_TEST_SET" = "true" ]; then
        run_test_set_evaluation "$provider" "$model" "$port" "$run_num"
        test_exit_code=$?
        
        # Evaluate test set results if enabled and inference succeeded
        if [ "$RUN_RESULTS_EVAL" = "true" ] && [ $test_exit_code -eq 0 ]; then
            evaluate_test_set_results "$provider" "$model" "$run_num"
            test_results_exit_code=$?
        fi
    fi
    
    # Run heldout set evaluation if enabled
    if [ "$RUN_HELDOUT_SET" = "true" ]; then
        run_heldout_set_evaluation "$provider" "$model" "$port" "$run_num"
        heldout_exit_code=$?
        
        # Evaluate heldout set results if enabled and inference succeeded
        if [ "$RUN_RESULTS_EVAL" = "true" ] && [ $heldout_exit_code -eq 0 ]; then
            evaluate_heldout_set_results "$provider" "$model" "$run_num"
            heldout_results_exit_code=$?
        fi
    fi
    
    # If only results evaluation is enabled (inference skipped), run results evaluation directly
    if [ "$RUN_RESULTS_EVAL" = "true" ] && [ "$RUN_TEST_SET" = "false" ] && [ "$RUN_HELDOUT_SET" = "false" ]; then
        # Evaluate test set results
        evaluate_test_set_results "$provider" "$model" "$run_num"
        test_results_exit_code=$?
        
        # Evaluate heldout set results
        evaluate_heldout_set_results "$provider" "$model" "$run_num"
        heldout_results_exit_code=$?
    fi
    
    # Return non-zero if any evaluation failed
    if [ $test_exit_code -ne 0 ] || [ $heldout_exit_code -ne 0 ] || \
       [ $test_results_exit_code -ne 0 ] || [ $heldout_results_exit_code -ne 0 ]; then
        return 1
    fi
    
    return 0
}

# Run evaluations for all provider-model pairs
total_pairs=${#PROVIDER_MODEL_PAIRS[@]}
current=0

echo "============================================================"
echo "EMBGuard Unified Batch Evaluation"
echo "============================================================"
echo "Total pairs: $total_pairs"
echo "Number of runs: $NUM_RUNS"
[ "$RUN_TEST_SET" = "true" ] && echo "Test Set: $TEST_SET (workers: $TEST_SET_NUM_WORKERS)"
[ "$RUN_HELDOUT_SET" = "true" ] && echo "Heldout Set: $HELDOUT_DATASET (workers: $HELDOUT_NUM_WORKERS)"
[ "$RUN_RESULTS_EVAL" = "true" ] && echo "Results Evaluation: Enabled (judge: $JUDGE_PROVIDER/$JUDGE_MODEL, workers: $RESULTS_NUM_WORKERS)"
echo "Few-shot: $USE_FEW_SHOT"
echo "Thinking: $USE_THINKING"
echo "Parallel mode: $PARALLEL_MODE (max jobs: $MAX_PARALLEL_JOBS)"
echo "============================================================"
echo ""

# ============================================
# Parallel execution mode
# ============================================
if [ "$PARALLEL_MODE" = "true" ]; then
    echo "Starting parallel execution..."
    echo ""
    
    # Create a temporary directory for logs
    LOG_DIR="${PROJECT_ROOT}/logs/evaluation_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$LOG_DIR"
    echo "Logs will be saved to: $LOG_DIR"
    echo ""
    
    # Array to store background job PIDs
    declare -a PIDS=()
    declare -a JOB_NAMES=()
    
    # Calculate total jobs (pairs * runs)
    total_jobs=$((total_pairs * NUM_RUNS))
    job_counter=0
    
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
        
        # Run NUM_RUNS times for each pair
        for run_num in $(seq 1 $NUM_RUNS); do
            job_counter=$((job_counter + 1))
            
            # Create a clean name for log file
            model_clean=$(echo "$model" | sed 's/\//_/g' | sed 's/:/_/g')
            log_file="${LOG_DIR}/${provider}_${model_clean}_port${port}_Run${run_num}.log"
            job_name="${provider}/${model}:${port} (Run $run_num)"
            
            echo "[$job_counter/$total_jobs] Starting: $job_name"
            echo "  Log: $log_file"
            
            # Run in background and save PID
            (
                run_all_evaluations "$provider" "$model" "$port" "$run_num"
            ) > "$log_file" 2>&1 &
            
            PIDS+=($!)
            JOB_NAMES+=("$job_name")
            
            # Limit parallel jobs if MAX_PARALLEL_JOBS > 0
            if [ "$MAX_PARALLEL_JOBS" -gt 0 ] && [ ${#PIDS[@]} -ge "$MAX_PARALLEL_JOBS" ]; then
                echo ""
                echo "Reached max parallel jobs ($MAX_PARALLEL_JOBS), waiting for one to complete..."
                wait -n  # Wait for any one job to complete
                # Remove completed jobs from arrays (simplified - just continue)
            fi
        done
    done
    
    echo ""
    echo "============================================================"
    echo "All jobs started! Waiting for completion..."
    echo "Total jobs: ${#PIDS[@]} (${total_pairs} pairs × ${NUM_RUNS} runs)"
    echo "============================================================"
    echo ""
    echo "You can monitor progress with:"
    echo "  tail -f ${LOG_DIR}/*.log"
    echo ""
    echo "Or check specific log:"
    echo "  ls -la ${LOG_DIR}/"
    echo ""
    
    # Wait for all background jobs to complete
    failed_jobs=0
    for i in "${!PIDS[@]}"; do
        pid=${PIDS[$i]}
        job_name=${JOB_NAMES[$i]}
        
        if wait "$pid"; then
            echo "✓ Completed: $job_name"
        else
            echo "✗ Failed: $job_name (check log for details)"
            failed_jobs=$((failed_jobs + 1))
        fi
    done
    
    echo ""
    echo "============================================================"
    echo "All parallel evaluations completed!"
    echo "============================================================"
    echo "Total: ${#PIDS[@]} jobs"
    echo "Succeeded: $((${#PIDS[@]} - failed_jobs))"
    echo "Failed: $failed_jobs"
    echo "Logs: $LOG_DIR"
    echo "============================================================"

# ============================================
# Sequential execution mode (original behavior)
# ============================================
else
    # Calculate total jobs (pairs * runs)
    total_jobs=$((total_pairs * NUM_RUNS))
    job_counter=0
    
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
        
        # Run NUM_RUNS times for each pair
        for run_num in $(seq 1 $NUM_RUNS); do
            job_counter=$((job_counter + 1))
            echo "[$job_counter/$total_jobs] Processing: $provider / $model (Run $run_num)"
            run_all_evaluations "$provider" "$model" "$port" "$run_num"
            echo ""
        done
    done

    echo "============================================================"
    echo "All evaluations completed!"
    echo "Total: $total_jobs jobs (${total_pairs} pairs × ${NUM_RUNS} runs)"
    echo "============================================================"
fi
