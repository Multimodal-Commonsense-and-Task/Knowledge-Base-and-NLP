#!/usr/bin/env python3
"""
Benchmark vLLM inference latency: seconds per sample on EMBGuardTest data.
Uses a single sample from EMBGuardTest (Hugging Face) to measure time per data point.

Usage:
  # vLLM server must be running (e.g. scripts/run_vllm4.sh). Then:
  python src/evals/benchmark_vllm_latency.py
  python src/evals/benchmark_vllm_latency.py --vllm-port 8003 --model Qwen/Qwen3-VL-4B-Instruct
  python src/evals/benchmark_vllm_latency.py --warmup 1   # 1 warmup then timed run
"""
import argparse
import sys
import time
from pathlib import Path

# Project root
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.config import get_config
from utils.path import get_project_path
from src.guardrail.guardrail import EMBGuard
from src.evals.utils import load_data, resolve_image
from src.evals.test_set_helpers import create_model_config
from utils.test_types import normalize_test_type_code, normalize_test_type_key, test_type_slug


def main():
    parser = argparse.ArgumentParser(
        description="Measure vLLM inference time per sample using 1 EMBGuardTest sample"
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default=None,
        help="Hugging Face dataset (default: from config common.test_set.causal_risky, e.g. EMBGuard/EMBGuardTest_v2)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="causal_risky",
        help="Dataset split to take 1 sample from (for example: causal_risky, selective_risky, decoupled_benign, absent_benign; default: causal_risky)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-4B-Instruct",
        help="vLLM model name (default: Qwen/Qwen3-VL-4B-Instruct)",
    )
    parser.add_argument(
        "--vllm-port",
        type=str,
        default=None,
        help="vLLM server port (default: from config vllm.base_url, e.g. 8000)",
    )
    parser.add_argument(
        "--use-few-shot",
        action="store_true",
        default=True,
        help="Use few-shot prompt (default: True)",
    )
    parser.add_argument(
        "--no-few-shot",
        action="store_true",
        help="Disable few-shot prompt",
    )
    parser.add_argument(
        "--use-thinking",
        action="store_true",
        default=False,
        help="Use thinking mode",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of warmup inferences before timing (default: 0)",
    )
    args = parser.parse_args()

    use_few_shot = getattr(args, "use_few_shot", True) and not args.no_few_shot
    split_key = normalize_test_type_key(args.split, strict=True)
    split_slug = test_type_slug(args.split)
    split = normalize_test_type_code(args.split, default=args.split, strict=True)

    # Resolve data source
    if args.data_source:
        data_source = args.data_source
    else:
        config = get_config()
        test_set = config.get("common", {}).get("test_set", {})
        data_source = test_set.get(
            args.split,
            test_set.get(
                split_slug,
                test_set.get(
                    split_key,
                    test_set.get(
                        split.lower(),
                        test_set.get("causal_risky", test_set.get("hr", "EMBGuard/EMBGuardTest_v2")),
                    ),
                ),
            ),
        )
    if not data_source:
        print("Error: No data source. Set --data-source or config common.test_set.causal_risky.", file=sys.stderr)
        sys.exit(1)

    # Load 1 sample
    print(f"Loading 1 sample from {data_source} (split={split})...")
    df, is_hf_dataset, csv_dir = load_data(data_source, split=split)
    if df is None or len(df) == 0:
        print("Error: No data loaded.", file=sys.stderr)
        sys.exit(1)

    row = df.iloc[0]
    row_dict = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    csv_dir_path = Path(csv_dir) if csv_dir else None
    image_path = resolve_image(row_dict, csv_dir_path, is_hf_dataset=is_hf_dataset)
    action = row_dict.get("Action", "") or row_dict.get("action", "")
    if not action:
        print("Error: No Action/action in sample.", file=sys.stderr)
        sys.exit(1)

    # vLLM model config
    model_config = create_model_config(
        "vllm",
        args.model,
        vllm_port=args.vllm_port,
    )
    guard = EMBGuard("vllm", model_config)

    # Warmup
    if args.warmup > 0:
        print(f"Warmup: {args.warmup} inference(s)...")
        for _ in range(args.warmup):
            guard.evaluate(
                action=action,
                image=str(image_path),
                use_few_shot=use_few_shot,
                use_thinking=args.use_thinking,
            )

    # Timed inference
    print("Running 1 inference (timed)...")
    start = time.perf_counter()
    result = guard.evaluate(
        action=action,
        image=str(image_path),
        use_few_shot=use_few_shot,
        use_thinking=args.use_thinking,
    )
    elapsed = time.perf_counter() - start

    print(f"\n=== vLLM latency (1 sample) ===")
    print(f"  Time: {elapsed:.4f} s")
    print(f"  Per sample: {elapsed:.4f} s")
    if result.get("usage"):
        u = result["usage"]
        print(f"  Tokens: prompt={u.get('prompt_tokens', 'N/A')}, completion={u.get('completion_tokens', 'N/A')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
