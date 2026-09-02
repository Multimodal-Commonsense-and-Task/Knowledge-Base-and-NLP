#!/bin/bash

export NUM_GPUS=1

START_TIME=`date +%Y%m%d-%H:%M:%S`

if [[ -v PARTITION ]]; then
    echo "Submit to $PARTITION"
fi

# Get script directory first
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Set PYTHONPATH to project root (EMBGuard/) to enable importing src modules
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# vLLM base URL mapping per model
declare -A VLLM_BASE_URL_MAP=(
    # ["OpenGVLab/InternVL3-1B-hf"]="http://127.0.0.1:8002/v1"
    # ["OpenGVLab/InternVL3-2B-hf"]="http://127.0.0.1:8003/v1"
    # ["Qwen/Qwen3-VL-2B-Instruct"]="http://127.0.0.1:8001/v1"
    # ["Qwen/Qwen3-VL-4B-Instruct"]="http://127.0.0.1:8000/v1"
    ["EMBGuard/EMBGuard-2B"]="http://127.0.0.1:8004/v1"
    ["EMBGuard/EMBGuard-4B"]="http://127.0.0.1:8005/v1"
)

# Configuration
GUARDRAIL_LIST=(
    # Open-source - InternVL
    #"OpenGVLab/InternVL3-1B-hf:vllm" \
    #"OpenGVLab/InternVL3-2B-hf:vllm" \
    
    # Open-source - Qwen
    #"Qwen/Qwen3-VL-2B-Instruct:vllm" \
    #"Qwen/Qwen3-VL-4B-Instruct:vllm" \
    
    # EMBGUARD
    "EMBGuard/EMBGuard-2B:vllm" \
    "EMBGuard/EMBGuard-4B:vllm" \
)

JUDGE_MODEL=gpt-4o
JUDGE_PROVIDER=openai
BENCHMARK_DIR=../data/images
TASK_LIST=../entrypoints/task_list.txt
BASE_OUTPUT_DIR=../results/risk_detection_results/$START_TIME

# Number of parallel workers for task processing (default: 1 for sequential)
# Recommended: 10-20 for API-based models (OpenAI/OpenRouter)
# Higher values may hit rate limits. Monitor for 429 errors.
NUM_WORKERS=5

PIDS=()
MODEL_TAGS=()

for GUARDRAIL_PAIR in "${GUARDRAIL_LIST[@]}"; do
    MODEL=${GUARDRAIL_PAIR%%:*}
    PROVIDER=${GUARDRAIL_PAIR##*:}
    MODEL_TAG=${MODEL//\//_}-${PROVIDER}
    OUTPUT_DIR="$BASE_OUTPUT_DIR/$MODEL_TAG"
    LOG_FILE=$OUTPUT_DIR/logs/exec_${START_TIME}_${MODEL_TAG}.log

    if [[ "$PROVIDER" == "vllm" ]]; then
        VLLM_BASE_URL="${VLLM_BASE_URL_MAP[$MODEL]:-}"
        if [[ -z "$VLLM_BASE_URL" ]]; then
            echo "Missing VLLM base URL mapping for model: $MODEL"
            exit 1
        fi
        export VLLM_BASE_URL
        echo "VLLM base URL: $VLLM_BASE_URL"
    fi

    # Create output directory (after cd to script dir)
    mkdir -p "$OUTPUT_DIR/logs"

    echo "Starting risk detection evaluation..."
    echo "Guardrail Model: $MODEL"
    echo "Guardrail Provider: $PROVIDER"
    echo "Judge Model: $JUDGE_MODEL"
    echo "Judge Provider: $JUDGE_PROVIDER"
    echo "Benchmark dir: $BENCHMARK_DIR"
    echo "Task list: $TASK_LIST"
    echo "Output dir: $OUTPUT_DIR"
    echo "Log file: $LOG_FILE"
    echo "Number of workers: $NUM_WORKERS"

    # Run evaluation in background
    python ../src/evaluator/risk_detection_evaluator.py \
        --task_list $TASK_LIST \
        --benchmark_dir $BENCHMARK_DIR \
        --output_dir $OUTPUT_DIR \
        --guardrail_model $MODEL \
        --guardrail_provider $PROVIDER \
        --judge_model $JUDGE_MODEL \
        --judge_provider $JUDGE_PROVIDER \
        --num_workers $NUM_WORKERS \
        2>&1 | tee -a "$LOG_FILE" > /dev/null &

    PYTHON_PID=$!
    echo "Started evaluation (PID: $PYTHON_PID)"
    PIDS+=("$PYTHON_PID")
    MODEL_TAGS+=("$MODEL_TAG")
done

# Wait for all evaluations to complete
EXIT_CODE=0
for i in "${!PIDS[@]}"; do
    PID="${PIDS[$i]}"
    TAG="${MODEL_TAGS[$i]}"
    wait "$PID"
    CODE=$?
    echo "Evaluation complete for $TAG. Exit code: $CODE"
    if [ "$CODE" -ne 0 ]; then
        EXIT_CODE=$CODE
    fi
done

exit $EXIT_CODE

