#!/usr/bin/env python3
"""
Unified evaluation entry point for EMBGuard
Supports test set, heldout set, and results evaluation
"""
import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from utils.config import get_config, load_config
from utils.path import get_project_path
from utils.test_types import TEST_TYPE_CODES, normalize_test_type_code, normalize_test_type_key, test_type_label
from src.evals.test_set_evaluator import TestSetEvaluator
from src.evals.heldout_set_evaluator import HeldoutSetEvaluator
from src.evals.results_evaluator import ResultsEvaluator


def create_model_config(provider: str, model_name: str, config_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """Create model configuration from config file and kwargs"""
    # Load config
    if config_path:
        config = load_config(config_path)
    else:
        config = get_config()
    
    # Configure model settings
    model_config = {
        "model_name": model_name,
        "temperature": kwargs.get("temperature", 0.7),
        "max_tokens": kwargs.get("max_tokens", 8192),
    }
    
    # Set API keys and base_url
    provider_key_map = {
        "openai": "openai",
        "openrouter": "openrouter",
        "claude": "anthropic",
        "gemini": "gemini",
        "vllm": "vllm",
    }
    
    config_key = provider_key_map.get(provider.lower())
    if config_key and config_key in config:
        if "key" in config[config_key]:
            model_config["api_key"] = config[config_key].get("key", "")
        if "base_url" in config[config_key]:
            model_config["base_url"] = config[config_key].get("base_url", "")
    
    # Handle vLLM port override
    if provider.lower() == "vllm" and "vllm_port" in kwargs and kwargs["vllm_port"]:
        port = kwargs["vllm_port"]
        base_url = model_config.get("base_url", "http://127.0.0.1:8000/v1")
        import re
        if re.match(r'^https?://', base_url):
            match = re.match(r'^(https?://)([^:/]+)(?::(\d+))?(/.*)?$', base_url)
            if match:
                protocol = match.group(1)
                host = match.group(2)
                path = match.group(4) or "/v1"
                model_config["base_url"] = f"{protocol}{host}:{port}{path}"
            else:
                model_config["base_url"] = f"http://127.0.0.1:{port}/v1"
        else:
            model_config["base_url"] = f"http://127.0.0.1:{port}/v1"
    
    # Additional settings (kwargs takes precedence)
    if "base_url" in kwargs and kwargs["base_url"]:
        model_config["base_url"] = kwargs["base_url"]
    
    return model_config


def run_test_set_evaluation(args):
    """Run test set evaluation"""
    model_config = create_model_config(
        args.provider,
        args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        base_url=args.base_url,
        vllm_port=args.vllm_port,
    )
    
    evaluator = TestSetEvaluator(args.provider, model_config, args.output_dir)
    
    # Get use_thinking from config if not provided
    config = get_config()
    common_config = config.get("common", {})
    default_use_thinking = common_config.get("use_thinking", False)
    use_thinking = args.use_thinking if hasattr(args, 'use_thinking') else default_use_thinking
    
    if args.data_source:
        split = normalize_test_type_code(args.split, default=args.split)
        test_set_type = normalize_test_type_code(args.test_set_type, default=args.test_set_type)
        # Single data source
        results = evaluator.run(
            data_source=args.data_source,
            split=split,
            use_few_shot=not args.no_few_shot,
            use_thinking=use_thinking,
            num_workers=args.num_workers,
            test_set_type=test_set_type,
        )
        print(f"\nGenerated {len(results)} results")
    else:
        # Multiple test sets from config
        from src.evals.test_set_helpers import get_test_set_paths, evaluate_test_sets
        test_set_paths = get_test_set_paths()
        
        test_set_arg = args.test_set.lower().strip()
        if test_set_arg == "all":
            test_sets = [code.lower() for code in TEST_TYPE_CODES]
        else:
            test_sets = [normalize_test_type_key(ts.strip(), strict=True) for ts in args.test_set.split(",")]
        
        all_results = evaluate_test_sets(
            provider=args.provider,
            model_name=args.model,
            test_sets=test_sets,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            output_dir=args.output_dir,
            use_few_shot=not args.no_few_shot,
            use_thinking=use_thinking,
            num_workers=args.num_workers,
            base_url=args.base_url,
            vllm_port=args.vllm_port,
        )
        
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for test_set_name, results in all_results.items():
            success_count = len([r for r in results if 'error' not in r])
            total_count = len(results)
            print(f"{test_type_label(test_set_name)}: {success_count}/{total_count} successful")


def run_heldout_set_evaluation(args):
    """Run heldout set evaluation"""
    model_config = create_model_config(
        args.provider,
        args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        base_url=args.base_url,
        vllm_port=args.vllm_port,
    )
    
    evaluator = HeldoutSetEvaluator(args.provider, model_config, args.output_dir)
    
    config = get_config()
    common_config = config.get("common", {})
    default_use_thinking = common_config.get("use_thinking", False)
    use_thinking = args.use_thinking if hasattr(args, 'use_thinking') else default_use_thinking
    
    if args.data_source:
        # Single data source
        results = evaluator.run(
            data_source=args.data_source,
            split=args.split,
            use_few_shot=not args.no_few_shot,
            use_thinking=use_thinking,
            num_workers=args.num_workers,
        )
        print(f"\nGenerated {len(results)} results")
    else:
        # Multiple datasets from config
        from src.evals.heldout_set_helpers import get_heldout_set_paths, evaluate_heldout_sets
        heldout_set_paths = get_heldout_set_paths()
        
        dataset_arg = args.dataset.lower().strip()
        if dataset_arg == "all":
            datasets = [code.lower() for code in TEST_TYPE_CODES]
        else:
            datasets = [
                normalize_test_type_key(ds.strip(), strict=False) or ds.strip().lower()
                for ds in dataset_arg.split(",")
            ]
        
        all_results = evaluate_heldout_sets(
            provider=args.provider,
            model_name=args.model,
            datasets=datasets,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            output_dir=args.output_dir,
            use_few_shot=not args.no_few_shot,
            use_thinking=use_thinking,
            num_workers=args.num_workers,
            base_url=args.base_url,
            vllm_port=args.vllm_port,
        )
        
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for dataset_name, results in all_results.items():
            success_count = len([r for r in results if 'error' not in r])
            total_count = len(results)
            print(f"{dataset_name.upper()}: {success_count}/{total_count} successful")


def run_results_evaluation(args):
    """Run results evaluation"""
    from src.evals.results_evaluator import get_output_dir_from_results_file
    
    judge_config = {
        "temperature": args.temperature if hasattr(args, 'temperature') else 0.0,
        "max_tokens": args.max_tokens if hasattr(args, 'max_tokens') else 512,
    }
    if hasattr(args, 'base_url') and args.base_url:
        judge_config["base_url"] = args.base_url
    
    evaluator = ResultsEvaluator(
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        judge_config=judge_config,
    )
    
    # Auto-generate output file if not provided
    output_file = args.output_file
    if not output_file:
        output_dir = get_output_dir_from_results_file(args.results_file)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = Path(args.results_file)
        results_filename = results_path.stem
        if results_filename.endswith("_results"):
            evaluation_filename = results_filename[:-8] + "_evaluation.json"
        else:
            evaluation_filename = results_filename + "_evaluation.json"
        output_file = str(output_dir / evaluation_filename)
    
    eval_output = evaluator.evaluate_results_file(
        results_file=args.results_file,
        output_file=output_file,
        num_workers=args.num_workers,
    )
    
    stats = eval_output["statistics"]
    print(f"\nEvaluation complete. Results saved to: {output_file}")
    print(f"\nOverall Accuracy: {stats['overall_accuracy']:.4f} ({stats['overall_accuracy']*100:.2f}%)")
    print(f"Potential Risk Accuracy: {stats['potential_risk_accuracy']:.4f} ({stats['potential_risk_accuracy']*100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="EMBGuard Unified Evaluation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test set evaluation
  python src/evaluate.py test-set --provider openai --model gpt-4o --test-set causal_risky
  
  # Heldout set evaluation
  python src/evaluate.py heldout-set --provider openai --model gpt-4o --dataset safe
  
  # Results evaluation
  python src/evaluate.py results --results-file outputs/.../results.jsonl --output-file results/.../evaluation.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Evaluation mode')
    
    # Test set evaluation
    test_parser = subparsers.add_parser('test-set', help='Evaluate on test set')
    test_parser.add_argument("--provider", type=str, required=True,
                           choices=["openai", "openrouter", "vllm", "claude", "gemini"],
                           help="LLM provider")
    test_parser.add_argument("--model", type=str, required=True, help="Model name")
    test_parser.add_argument("--test-set", type=str, default="all",
                           help="Test sets: 'all', 'causal_risky', 'selective_risky', 'decoupled_benign', 'absent_benign', or comma-separated")
    test_parser.add_argument("--data-source", "--csv", type=str, default=None,
                           dest="data_source",
                           help="CSV file path or Hugging Face dataset name")
    test_parser.add_argument("--split", type=str, default=None,
                           help="Split name for Hugging Face dataset")
    test_parser.add_argument("--test-set-type", type=str, default=None,
                           help="Test set type label or alias (for example: causal_risky, selective_risky, Causal Risky)")
    test_parser.add_argument("--temperature", type=float, default=0.7)
    test_parser.add_argument("--max_tokens", type=int, default=2048)
    test_parser.add_argument("--output_dir", type=str, default=None)
    test_parser.add_argument("--no-few-shot", action="store_true")
    test_parser.add_argument("--use-thinking", action="store_true")
    test_parser.add_argument("--num-workers", type=int, default=1)
    test_parser.add_argument("--base-url", type=str, default=None)
    test_parser.add_argument("--vllm-port", type=str, default=None)
    
    # Heldout set evaluation
    heldout_parser = subparsers.add_parser('heldout-set', help='Evaluate on heldout set')
    heldout_parser.add_argument("--provider", type=str, required=True,
                              choices=["openai", "openrouter", "vllm", "claude", "gemini"],
                              help="LLM provider")
    heldout_parser.add_argument("--model", type=str, required=True, help="Model name")
    heldout_parser.add_argument("--dataset", type=str, default="all",
                              help="Datasets: 'all', 'causal_risky', 'selective_risky', 'decoupled_benign', 'absent_benign', 'safe', 'unsafe', or comma-separated")
    heldout_parser.add_argument("--data-source", "--csv", type=str, default=None,
                              dest="data_source",
                              help="CSV file path or Hugging Face dataset name")
    heldout_parser.add_argument("--split", type=str, default=None,
                              help="Split name for Hugging Face dataset")
    heldout_parser.add_argument("--temperature", type=float, default=0.7)
    heldout_parser.add_argument("--max_tokens", type=int, default=2048)
    heldout_parser.add_argument("--output_dir", type=str, default=None)
    heldout_parser.add_argument("--no-few-shot", action="store_true")
    heldout_parser.add_argument("--use-thinking", action="store_true")
    heldout_parser.add_argument("--num-workers", type=int, default=1)
    heldout_parser.add_argument("--base-url", type=str, default=None)
    heldout_parser.add_argument("--vllm-port", type=str, default=None)
    
    # Results evaluation
    results_parser = subparsers.add_parser('results', help='Evaluate results file')
    results_parser.add_argument("--results-file", type=str, required=True,
                              help="Path to JSONL results file")
    results_parser.add_argument("--output-file", type=str, default=None,
                              help="Path to output evaluation JSON file (auto-generated if not provided)")
    results_parser.add_argument("--judge-provider", type=str, default="openai",
                              choices=["openai", "openrouter", "vllm", "claude", "gemini"],
                              help="Judge model provider")
    results_parser.add_argument("--judge-model", type=str, default="gpt-4o",
                              help="Judge model name")
    results_parser.add_argument("--num-workers", type=int, default=16,
                              help="Number of worker processes")
    results_parser.add_argument("--temperature", type=float, default=0.0)
    results_parser.add_argument("--max_tokens", type=int, default=512)
    results_parser.add_argument("--base-url", type=str, default=None)
    
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        sys.exit(1)
    
    if args.mode == 'test-set':
        run_test_set_evaluation(args)
    elif args.mode == 'heldout-set':
        run_heldout_set_evaluation(args)
    elif args.mode == 'results':
        run_results_evaluation(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
