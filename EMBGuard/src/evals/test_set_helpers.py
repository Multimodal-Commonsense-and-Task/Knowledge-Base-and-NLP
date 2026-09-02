"""
Helper functions for test set evaluation
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

from utils.config import get_config, load_config
from utils.path import get_project_path
from utils.test_types import TEST_TYPE_CODES, normalize_test_type_code, normalize_test_type_key, test_type_label, test_type_slug
from src.evals.test_set_evaluator import TestSetEvaluator


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


def evaluate_from_config(
    provider: str,
    model_name: str,
    data_source: str,
    config_path: Optional[str] = None,
    test_set_type: Optional[str] = None,
    split: Optional[str] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Run evaluation from config file (convenience function)
    
    Args:
        provider: LLM provider
        model_name: Model name
        data_source: CSV file path or Hugging Face dataset name (e.g., "org/dataset_name")
        config_path: Config file path (uses default if None)
        test_set_type: Test set label or alias (for example, "causal_risky"
            or "Causal Risky"). Auto-detected if None.
        split: Split name for Hugging Face dataset (for example, "causal_risky")
        **kwargs: Additional model settings (temperature, max_tokens, use_thinking, etc.)
        
    Returns:
        List of evaluation results
    """
    config = load_config(config_path) if config_path else get_config()
    common_config = config.get("common", {})
    default_use_thinking = common_config.get("use_thinking", False)
    
    model_config = create_model_config(provider, model_name, config_path, **kwargs)
    
    evaluator = TestSetEvaluator(provider, model_config, kwargs.get("output_dir"))
    return evaluator.run(
        data_source=data_source,
        output_filename=kwargs.get("output_filename"),
        save_intermediate=kwargs.get("save_intermediate", True),
        use_few_shot=kwargs.get("use_few_shot", True),
        use_thinking=kwargs.get("use_thinking", default_use_thinking),
        num_workers=kwargs.get("num_workers", 1),
        test_set_type=test_set_type,
        split=split,
    )


def get_test_set_paths(config_path: Optional[str] = None) -> Dict[str, str]:
    """
    Get test set CSV file paths or Hugging Face dataset names from config
    
    Args:
        config_path: Config file path (uses default if None)
        
    Returns:
        Dictionary mapping normalized test set names to file paths or HF dataset names
        - If path contains "/" and doesn't exist as file, treated as Hugging Face dataset name
        - Otherwise, treated as local CSV file path
    """
    config = load_config(config_path) if config_path else get_config()
    common_config = config.get("common", {})
    test_set_config = common_config.get("test_set", {})
    if not test_set_config:
        raise ValueError("test_set configuration not found in config file (under common section)")
    
    project_path = get_project_path()
    test_set_paths = {}
    for raw_key, path_or_dataset in test_set_config.items():
        key = normalize_test_type_key(raw_key)
        if key not in [code.lower() for code in TEST_TYPE_CODES]:
            continue

        # Check if it's a Hugging Face dataset name (contains "/" and doesn't exist as file)
        if "/" in path_or_dataset and not Path(path_or_dataset).exists():
            # Treat as Hugging Face dataset name
            test_set_paths[key] = path_or_dataset
        else:
            # Treat as local CSV file path
            if not Path(path_or_dataset).is_absolute():
                path = project_path / path_or_dataset
            else:
                path = Path(path_or_dataset)
            test_set_paths[key] = str(path)
    
    return test_set_paths


def evaluate_test_sets(
    provider: str,
    model_name: str,
    test_sets: List[str],
    config_path: Optional[str] = None,
    **kwargs
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Evaluate specified test sets
    
    Args:
        provider: LLM provider
        model_name: Model name
        test_sets: List of test set names to evaluate (for example,
            ["causal_risky", "decoupled_benign"])
        config_path: Config file path (uses default if None)
        **kwargs: Additional model settings
        
    Returns:
        Dictionary mapping test set names to results
    """
    test_set_paths = get_test_set_paths(config_path)
    
    valid_test_sets = [code.lower() for code in TEST_TYPE_CODES]
    normalized_test_sets = [normalize_test_type_key(ts, strict=True) for ts in test_sets]
    invalid_sets = [ts for ts in normalized_test_sets if ts not in valid_test_sets]
    if invalid_sets:
        raise ValueError(f"Invalid test set names: {invalid_sets}. Valid names: {valid_test_sets}")
    
    all_results = {}
    for test_set_lower in normalized_test_sets:
        if test_set_lower not in test_set_paths:
            print(f"Warning: Test set '{test_set_lower}' not found in config, skipping...")
            continue
        
        data_source = test_set_paths[test_set_lower]
        
        # Determine if it's a Hugging Face dataset or local CSV
        is_hf_dataset = "/" in data_source and not Path(data_source).exists()
        if is_hf_dataset:
            split = test_type_slug(test_set_lower)
            print(f"\n{'='*60}")
            print(f"Evaluating: {test_type_label(test_set_lower)} (Hugging Face: {data_source}, split: {split})")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print(f"Evaluating: {test_type_label(test_set_lower)} (Local CSV: {data_source})")
            print(f"{'='*60}")
            split = None
        
        try:
            test_set_type_upper = normalize_test_type_code(test_set_lower)
            results = evaluate_from_config(
                provider=provider,
                model_name=model_name,
                data_source=data_source,
                config_path=config_path,
                test_set_type=test_set_type_upper,
                split=split,  # Pass split for Hugging Face datasets
                **kwargs
            )
            all_results[test_set_lower] = results
        except Exception as e:
            print(f"Error evaluating {test_set_lower}: {e}")
            all_results[test_set_lower] = []
    
    return all_results
