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

# Configuration
GUARDRAIL_LIST=(
    # Closed-source (via OpenRouter)
    #"gpt-4o-mini:openai" \
    #"gpt-4o:openai" \
    "gpt-5.1:openai" \
    #"google/gemini-2.5-flash:openrouter" \
    "google/gemini-2.5-pro:openrouter" \
    
    # Open-source - Qwen
    #"qwen/qwen3-vl-8b-instruct:openrouter" \
    #"qwen/qwen3-vl-30b-a3b-instruct:openrouter" \
    "qwen/qwen3-vl-32b-instruct:openrouter" \
    #"qwen/qwen3-vl-235b-a22b-instruct:openrouter" \
    
    # Open-source - Gemma
    #"google/gemma-3-4b-it:openrouter" \
    #"google/gemma-3-12b-it:openrouter" \
    #"google/gemma-3-27b-it:openrouter" \
)

JUDGE_MODEL=gpt-4o
JUDGE_PROVIDER=openai
# Images are stored under IS-Bench/data/images relative to this script
BENCHMARK_DIR=../data/images
TASK_LIST=../entrypoints/task_list.txt
BASE_OUTPUT_DIR=../results/risk_detection_results/$START_TIME

# Number of parallel workers for task processing (default: 1 for sequential)
# Recommended: 10-20 for API-based models (OpenAI/OpenRouter)
# Higher values may hit rate limits. Monitor for 429 errors.
NUM_WORKERS=20

for GUARDRAIL_PAIR in "${GUARDRAIL_LIST[@]}"; do
    MODEL=${GUARDRAIL_PAIR%%:*}
    PROVIDER=${GUARDRAIL_PAIR##*:}
    MODEL_TAG=${MODEL//\//_}-${PROVIDER}
    OUTPUT_DIR="$BASE_OUTPUT_DIR/$MODEL_TAG"
    LOG_FILE=$OUTPUT_DIR/logs/exec_${START_TIME}_${MODEL_TAG}.log

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

    # Run evaluation
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
    sleep 0.5s
    tail -f $LOG_FILE &
    TAIL_PID=$!

    # Wait for Python process to complete
    wait $PYTHON_PID
    PYTHON_EXIT_CODE=$?

    # Kill tail process
    kill $TAIL_PID 2>/dev/null

    echo "Evaluation complete. Exit code: $PYTHON_EXIT_CODE"
    echo "Results saved to: $OUTPUT_DIR"

    if [ "$PYTHON_EXIT_CODE" -ne 0 ]; then
        exit $PYTHON_EXIT_CODE
    fi
done

exit 0