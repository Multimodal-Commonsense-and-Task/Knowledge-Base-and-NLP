"""
Helper functions for heldout set evaluation
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

from utils.config import get_config, load_config
from utils.path import get_project_path
from utils.test_types import TEST_TYPE_CODES, normalize_test_type_key, test_type_label, test_type_slug
from src.evals.heldout_set_evaluator import HeldoutSetEvaluator


SAFETY_HELDOUT_SPLITS = ("safe", "unsafe")
TYPE_HELDOUT_KEYS = tuple(code.lower() for code in TEST_TYPE_CODES)


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
        split: Split name for Hugging Face dataset (for example, "causal_risky" or "safe")
        **kwargs: Additional model settings (temperature, max_tokens, use_thinking, etc.)
        
    Returns:
        List of evaluation results
    """
    config = load_config(config_path) if config_path else get_config()
    common_config = config.get("common", {})
    default_use_thinking = common_config.get("use_thinking", False)
    
    model_config = create_model_config(provider, model_name, config_path, **kwargs)
    
    evaluator = HeldoutSetEvaluator(provider, model_config, kwargs.get("output_dir"))
    return evaluator.run(
        data_source=data_source,
        output_filename=kwargs.get("output_filename"),
        save_intermediate=kwargs.get("save_intermediate", True),
        use_few_shot=kwargs.get("use_few_shot", True),
        use_thinking=kwargs.get("use_thinking", default_use_thinking),
        num_workers=kwargs.get("num_workers", 1),
        split=split,
    )


def get_heldout_set_paths(config_path: Optional[str] = None) -> Dict[str, str]:
    """
    Get heldout set CSV file paths from config
    
    Args:
        config_path: Config file path (uses default if None)
        
    Returns:
        Dictionary mapping heldout split names to file paths or Hugging Face dataset names
    """
    config = load_config(config_path) if config_path else get_config()
    common_config = config.get("common", {})
    heldout_set_config = common_config.get("heldout_set", {})
    if not heldout_set_config:
        raise ValueError("heldout_set configuration not found in config file (under common section)")
    
    project_path = get_project_path()
    heldout_set_paths = {}
    for raw_key, path in heldout_set_config.items():
        key_lower = str(raw_key).strip().lower()
        key = key_lower if key_lower in SAFETY_HELDOUT_SPLITS else normalize_test_type_key(raw_key)
        if key not in SAFETY_HELDOUT_SPLITS and key not in TYPE_HELDOUT_KEYS:
            continue

        # Check if it's a Hugging Face dataset (contains "/" and doesn't exist as file)
        if "/" in path and not Path(path).exists():
            # Hugging Face dataset name - keep as is
            heldout_set_paths[key] = path
        else:
            # Local CSV file path
            if not Path(path).is_absolute():
                path = project_path / path
            else:
                path = Path(path)
            heldout_set_paths[key] = str(path)
    
    return heldout_set_paths


def evaluate_heldout_sets(
    provider: str,
    model_name: str,
    datasets: List[str],
    config_path: Optional[str] = None,
    **kwargs
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Evaluate specified heldout sets
    
    Args:
        provider: LLM provider
        model_name: Model name
        datasets: List of dataset names to evaluate (for example,
            ["causal_risky", "decoupled_benign"] or ["safe", "unsafe"])
        config_path: Config file path (uses default if None)
        **kwargs: Additional model settings
        
    Returns:
        Dictionary mapping dataset names to results
    """
    heldout_set_paths = get_heldout_set_paths(config_path)
    
    valid_datasets = list(TYPE_HELDOUT_KEYS) + list(SAFETY_HELDOUT_SPLITS)
    normalized_datasets = []
    for dataset in datasets:
        dataset_lower = dataset.lower()
        normalized = dataset_lower if dataset_lower in SAFETY_HELDOUT_SPLITS else normalize_test_type_key(dataset, strict=True)
        normalized_datasets.append(normalized)

    invalid_datasets = [ds for ds in normalized_datasets if ds not in valid_datasets]
    if invalid_datasets:
        raise ValueError(f"Invalid dataset names: {invalid_datasets}. Valid names: {valid_datasets}")
    
    all_results = {}
    for dataset_lower in normalized_datasets:
        if dataset_lower not in heldout_set_paths:
            print(f"Warning: Dataset '{dataset_lower}' not found in config, skipping...")
            continue
        
        data_source = heldout_set_paths[dataset_lower]
        
        # Determine if it's a Hugging Face dataset or local CSV
        is_hf_dataset = "/" in data_source and not Path(data_source).exists()
        display_name = dataset_lower.upper() if dataset_lower in SAFETY_HELDOUT_SPLITS else test_type_label(dataset_lower)
        if is_hf_dataset:
            split = dataset_lower if dataset_lower in SAFETY_HELDOUT_SPLITS else test_type_slug(dataset_lower)
            print(f"\n{'='*60}")
            print(f"Evaluating: {display_name} (Hugging Face: {data_source}, split: {split})")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print(f"Evaluating: {display_name} (Local CSV: {data_source})")
            print(f"{'='*60}")
            split = None
        
        try:
            results = evaluate_from_config(
                provider=provider,
                model_name=model_name,
                data_source=data_source,
                config_path=config_path,
                split=split,
                **kwargs
            )
            all_results[dataset_lower] = results
        except Exception as e:
            print(f"Error evaluating {dataset_lower}: {e}")
            all_results[dataset_lower] = []
    
    return all_results
