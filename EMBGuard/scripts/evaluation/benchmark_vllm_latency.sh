#!/bin/bash

# vLLM latency benchmark: seconds per sample on EMBGuardTest (1 sample)
# Usage: bash scripts/evaluation/benchmark_vllm_latency.sh
# Ensure vLLM server is running (e.g. scripts/run_vllm4.sh) before running.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# ============================================
# Configuration
# ============================================
VLLM_PORT="8000"                    # vLLM server port (match run_vllm*.sh)
MODEL="EMBGuard/EMBGuard-2B"       # Model name
SPLIT="causal_risky"               # EMBGuardTest split name
WARMUP=1                            # Warmup inferences before timing (0 = none)
# ============================================

echo "============================================================"
echo "vLLM Latency Benchmark (1 sample from EMBGuardTest)"
echo "============================================================"
echo "Port: $VLLM_PORT"
echo "Model: $MODEL"
echo "Split: $SPLIT"
echo "Warmup: $WARMUP"
echo "============================================================"
echo ""

python src/evals/benchmark_vllm_latency.py \
    --vllm-port "$VLLM_PORT" \
    --model "$MODEL" \
    --split "$SPLIT" \
    --warmup "$WARMUP" \
    "$@"

exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "Benchmark failed with exit code $exit_code"
    exit $exit_code
fi
echo ""
echo "Done."
