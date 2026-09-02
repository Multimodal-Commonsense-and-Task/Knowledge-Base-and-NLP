"""
Heldout set evaluator for EMBGuard
Evaluates models on heldout set CSV files or Hugging Face datasets
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm
import pandas as pd
import argparse
from datasets import load_dataset

from utils.config import get_config
from utils.path import get_project_path
from utils.test_types import hf_split_candidates, normalize_test_type_code, test_type_safety_type, test_type_slug
from src.guardrail.guardrail import EMBGuard
from src.inference_utils import run_parallel_inference
from src.evals.utils import load_data, resolve_image, convert_messages_for_storage


class HeldoutSetEvaluator:
    """Evaluator for heldout set CSV files"""
    
    def __init__(
        self,
        provider: str,
        model_config: Dict[str, Any],
        output_dir: Optional[str] = None,
    ):
        """
        Args:
            provider: LLM provider ("openai", "openrouter", "vllm", "claude", "gemini")
            model_config: Model configuration dictionary
            output_dir: Output directory for results (if provided, overrides default folder)
        """
        self.provider = provider
        self.model_config = model_config
        self.guard = EMBGuard(provider, model_config)
        # If provided, this overrides the default outputs/heldout_set/{provider}_{model_name}/
        self.output_dir = output_dir
    
    def _load_data(self, data_source: str, split: Optional[str] = None):
        """Load data using shared utility"""
        return load_data(data_source, split=split)
    
    def load_data(self, data_source: str, split: Optional[str] = None) -> tuple[pd.DataFrame, bool, Optional[Path]]:
        """
        Load data from CSV file or Hugging Face dataset
        
        Args:
            data_source: Path to CSV file or Hugging Face dataset name (e.g., "org/dataset_name")
            split: Split name for Hugging Face dataset (e.g., "safe", "unsafe")
            
        Returns:
            Tuple of (DataFrame, is_hf_dataset, csv_dir)
            - DataFrame: DataFrame with test data
            - is_hf_dataset: True if loaded from Hugging Face, False if from CSV
            - csv_dir: Directory containing CSV (None for HF datasets)
        """
        # Check if it's a Hugging Face dataset (contains "/" and doesn't exist as file)
        if "/" in data_source and not Path(data_source).exists():
            # Try to load from Hugging Face Hub
            try:
                # Suppress verbose output from datasets library (including image loading messages)
                import os
                import warnings
                old_verbosity = os.environ.get("HF_DATASETS_VERBOSITY", None)
                os.environ["HF_DATASETS_VERBOSITY"] = "error"
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if split:
                        last_error = None
                        for split_candidate in hf_split_candidates(split):
                            try:
                                dataset = load_dataset(data_source, split=split_candidate)
                                split = split_candidate
                                break
                            except Exception as e:
                                last_error = e
                        else:
                            raise last_error
                    else:
                        # Load all splits and combine
                        dataset_dict = load_dataset(data_source)
                        # If multiple splits, use the first one or specified split
                        if isinstance(dataset_dict, dict):
                            if split and split in dataset_dict:
                                dataset = dataset_dict[split]
                            else:
                                # Use first available split
                                split_name = list(dataset_dict.keys())[0]
                                dataset = dataset_dict[split_name]
                        else:
                            dataset = dataset_dict
                        
                        # Convert to pandas DataFrame (suppress image loading messages)
                        # Note: to_pandas() may convert PIL Images to strings, so we need to handle this
                        df = dataset.to_pandas()
                        
                        # If "image" column exists, try to restore PIL Images from strings
                        if "image" in df.columns:
                            from PIL import Image as PILImage
                            import io
                            
                            def restore_image(img_value):
                                """Restore PIL Image from dict/string/bytes or keep as PIL Image"""
                                if img_value is None or pd.isna(img_value):
                                    return None
                                # If already PIL Image, return as is
                                if isinstance(img_value, PILImage.Image):
                                    return img_value
                                # If dict (from to_pandas conversion), extract bytes or path
                                if isinstance(img_value, dict):
                                    if "bytes" in img_value and img_value["bytes"]:
                                        # Use bytes if available
                                        return PILImage.open(io.BytesIO(img_value["bytes"])).convert("RGB")
                                    elif "path" in img_value and img_value["path"]:
                                        # Fallback to path if bytes not available
                                        try:
                                            return PILImage.open(img_value["path"]).convert("RGB")
                                        except:
                                            return None
                                    else:
                                        return None
                                # If string (base64 or bytes), try to convert back
                                if isinstance(img_value, str):
                                    try:
                                        # Try to decode as base64
                                        import base64
                                        img_bytes = base64.b64decode(img_value)
                                        return PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
                                    except:
                                        # If not base64, try to open as file path
                                        try:
                                            return PILImage.open(img_value).convert("RGB")
                                        except:
                                            # If that fails, return None
                                            return None
                                # If bytes, convert to PIL Image
                                if isinstance(img_value, bytes):
                                    return PILImage.open(io.BytesIO(img_value)).convert("RGB")
                                return img_value
                            
                            # Apply restoration to image column
                            df["image"] = df["image"].apply(restore_image)
                
                # Restore verbosity
                if old_verbosity:
                    os.environ["HF_DATASETS_VERBOSITY"] = old_verbosity
                elif "HF_DATASETS_VERBOSITY" in os.environ:
                    del os.environ["HF_DATASETS_VERBOSITY"]
                
                return df, True, None
            except Exception as e:
                raise ValueError(f"Failed to load Hugging Face dataset {data_source}: {e}")
        else:
            # Load from CSV file
            csv_path = Path(data_source)
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
            df = pd.read_csv(csv_path)
            return df, False, csv_path.parent
    
    def load_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Load CSV file (backward compatibility)
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            DataFrame with test data
        """
        df, _, _ = self.load_data(csv_path)
        return df
    
    def _convert_messages_for_storage(self, messages: List[Dict[str, Any]], image_path: str) -> List[Dict[str, Any]]:
        """
        Convert messages for storage by replacing image objects with image_path
        
        Args:
            messages: Original messages (may contain image objects or base64)
            image_path: Image path to use in stored messages
            
        Returns:
            Messages suitable for storage (with image_path instead of base64/image objects)
        """
        messages_copy = []
        for msg in messages:
            msg_copy = msg.copy()
            
            # If message has images, replace with image_path reference
            if "images" in msg_copy:
                # Replace images list with image_path reference
                msg_copy["images"] = [image_path]
            elif isinstance(msg_copy.get("content"), list):
                # Handle multimodal content (list format)
                content_copy = []
                for item in msg_copy["content"]:
                    if isinstance(item, dict):
                        if item.get("type") == "image_url":
                            # Replace base64 image_url with path reference
                            content_copy.append({
                                "type": "image_url",
                                "image_url": {"url": f"<image_path:{image_path}>", "detail": "low"}
                            })
                        else:
                            content_copy.append(item)
                    else:
                        content_copy.append(item)
                msg_copy["content"] = content_copy
            
            messages_copy.append(msg_copy)
        
        return messages_copy
    
    def _resolve_image(self, item: Dict[str, Any], csv_dir: Optional[Path], is_hf_dataset: bool) -> Union[str, Path]:
        """
        Resolve image from image_path or Hugging Face dataset
        
        Args:
            item: Row/item dictionary containing image information
            csv_dir: Directory containing the CSV file (None for HF datasets)
            is_hf_dataset: Whether this is from Hugging Face dataset
            
        Returns:
            Image path (str or Path) or PIL Image object
        """
        if is_hf_dataset:
            # For Hugging Face datasets, check for "image" column first
            if "image" in item and item["image"] is not None:
                # PIL Image object from Hugging Face
                from PIL import Image as PILImage
                import tempfile
                
                pil_image = item["image"]
                # Create temporary file
                temp_dir = Path(tempfile.gettempdir()) / "embguard_images"
                temp_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate filename from ID or index
                image_id = item.get("id") or item.get("ID", f"img_{item.get('idx', 'unknown')}")
                temp_path = temp_dir / f"{image_id}.jpg"
                
                # Save PIL image to temp file
                if isinstance(pil_image, PILImage.Image):
                    pil_image.save(temp_path, "JPEG")
                    return temp_path
                else:
                    return str(pil_image)
            else:
                image_path_value = (
                    item.get("image_path", "")
                    or item.get("source_path", "")
                    or item.get("URL", "")
                    or item.get("url", "")
                )
                if image_path_value:
                    if Path(image_path_value).exists():
                        return Path(image_path_value)
                    return image_path_value
                raise ValueError("No image found in Hugging Face dataset item")
        else:
            # For CSV files, resolve image path
            image_path_value = (
                item.get("image_path", "")
                or item.get("source_path", "")
                or item.get("URL", "")
                or item.get("url", "")
                or item.get("path", "")
            )
            if not image_path_value:
                raise ValueError("No image path found in CSV row")
            
            # Image paths in CSV are relative to data/heldout_set
            if csv_dir:
                image_path = csv_dir / image_path_value
                if not image_path.exists():
                    raise FileNotFoundError(f"Image not found: {image_path_value} (tried: {image_path})")
                return image_path
            else:
                raise ValueError("csv_dir is required for CSV file images")
    
    def _resolve_image_path(self, image_path_value: str, csv_dir: Path) -> Path:
        """
        Resolve image path using heldout_set directory
        
        Args:
            image_path_value: Image path from CSV or dataset metadata
            csv_dir: Directory containing the CSV file (data/heldout_set)
            
        Returns:
            Resolved image path
        """
        item = {"image_path": image_path_value}
        result = self._resolve_image(item, csv_dir, is_hf_dataset=False)
        if isinstance(result, Path):
            return result
        return Path(result)
    
    def run(
        self,
        data_source: str,
        output_filename: Optional[str] = None,
        save_intermediate: bool = True,
        use_few_shot: bool = True,
        use_thinking: bool = False,
        num_workers: int = 1,
        split: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run evaluation on CSV file or Hugging Face dataset
        
        Args:
            data_source: Path to CSV file or Hugging Face dataset name (e.g., "org/dataset_name")
            output_filename: Output filename (auto-generated if None)
            save_intermediate: Whether to save intermediate results
            use_few_shot: Whether to include few-shot examples in prompts
            use_thinking: Whether to use thinking mode (step-by-step reasoning)
            num_workers: Number of worker processes (1 = sequential, >1 = parallel)
            split: Split name for Hugging Face dataset (e.g., "safe", "unsafe")
            
        Returns:
            List of evaluation results
        """
        # Load data (CSV or Hugging Face dataset)
        df, is_hf_dataset, csv_dir = load_data(data_source, split=split)
        
        # Extract dataset type from filename or split (safe or unsafe)
        if is_hf_dataset:
            # For HF datasets, use split name
            dataset_type = split or "unknown"
            csv_name = f"{data_source.replace('/', '_')}_{split}" if split else data_source.replace("/", "_")
        else:
            # For CSV files, extract from filename
            csv_path_obj = Path(data_source)
            csv_name = csv_path_obj.stem  # e.g., "dataset_safe" or "dataset_unsafe"
            dataset_type = "unknown"
            if "safe" in csv_name.lower():
                dataset_type = "safe"
            elif "unsafe" in csv_name.lower():
                dataset_type = "unsafe"
        
        # Set output directory
        # Use custom output_dir if provided, otherwise use default: outputs/heldout_set/{provider}_{model_name}
        if self.output_dir:
            dataset_output_dir = Path(self.output_dir)
        else:
            project_path = get_project_path()
            model_name = self.model_config.get("model_name", "model")
            model_name_clean = model_name.replace("/", "_").replace("\\", "_")
            provider_model_name = f"{self.provider}_{model_name_clean}"
            
            dataset_output_dir = project_path / "outputs" / "heldout_set" / provider_model_name
        
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get model name for filename generation
        model_name = self.model_config.get("model_name", "model")
        model_name_clean = model_name.replace("/", "_").replace("\\", "_")
        provider_model_name = f"{self.provider}_{model_name_clean}"
        
        # Set output file path
        if output_filename is None:
            # Build condition suffix
            few_shot_suffix = "few-shot" if use_few_shot else "no-few-shot"
            thinking_suffix = "thinking" if use_thinking else "non-thinking"
            condition_suffix = f"{few_shot_suffix}_{thinking_suffix}"
            
            output_filename = f"{self.provider}_{model_name_clean}_{csv_name}_{condition_suffix}_results.jsonl"
        
        output_path = dataset_output_dir / output_filename
        
        # Prepare dataset for evaluation
        dataset = []
        for idx, row in df.iterrows():
            row_dict = row.to_dict() if hasattr(row, 'to_dict') else row
            
            # Check for image (HF dataset has "image"; CSV/HF fallback has image_path).
            has_image = False
            image_path_value = (
                row_dict.get("image_path", "")
                or row_dict.get("source_path", "")
                or row_dict.get("URL", "")
                or row_dict.get("url", "")
                or row_dict.get("path", "")
            )
            if is_hf_dataset:
                if "image" in row_dict and row_dict["image"] is not None:
                    has_image = True
                elif image_path_value:
                    has_image = True
            else:
                if image_path_value and not (pd.isna(image_path_value) or image_path_value == ""):
                    has_image = True
            
            if not has_image:
                continue  # Skip rows without image
            
            # Get EMBGuardTest scenario type from normalized schema first, then legacy columns.
            scenario_code = "UNKNOWN"
            for type_column in ("scenario_type", "Type", "Subtype"):
                if type_column in row_dict:
                    scenario_code = normalize_test_type_code(row_dict.get(type_column, ""), default="UNKNOWN")
                    if scenario_code != "UNKNOWN":
                        break
            safety_type = str(row_dict.get("type") or dataset_type or "").strip().lower()
            if safety_type not in {"safe", "unsafe"}:
                safety_type = test_type_safety_type(scenario_code) or "unknown"
            
            dataset.append({
                "idx": int(idx),
                "row": row_dict,
                "image_path": image_path_value,
                "csv_dir": str(csv_dir) if csv_dir else None,
                "type": safety_type,
                "scenario_type": test_type_slug(scenario_code),
                "is_hf_dataset": is_hf_dataset,
            })
        
        # Prepare config for workers
        worker_config = {
            "provider": self.provider,
            "model_config": self.model_config,
            "use_few_shot": use_few_shot,
            "use_thinking": use_thinking,
        }
        
        # Run evaluation (sequential or parallel)
        if num_workers == 1:
            # Sequential evaluation
            results = []
            total_cost = 0.0
            total_usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            
            print(f"Starting evaluation on {len(dataset)} items...")
            for item in tqdm(dataset, desc="Evaluating"):
                try:
                    # Extract action and image path from row
                    row_dict = item["row"]
                    action = row_dict.get("action", "") or row_dict.get("Action", "")
                    if not action or action == "":
                        raise ValueError("Action field is missing or empty")
                    
                    # Resolve image (handles both CSV paths and HF dataset images)
                    is_hf = item.get("is_hf_dataset", False)
                    csv_dir_path = Path(csv_dir) if csv_dir else None
                    image_path = resolve_image(row_dict, csv_dir_path, is_hf_dataset=is_hf)
                    
                    # Prepare messages (to save in output)
                    messages = self.guard.prepare_messages(
                        action=action,
                        image=str(image_path),
                        use_few_shot=use_few_shot,
                        use_thinking=use_thinking
                    )
                    
                    # Convert messages for storage (replace image objects with image_path)
                    messages_for_storage = convert_messages_for_storage(messages, str(image_path))
                    
                    # Evaluate using EMBGuard
                    evaluation_result = self.guard.evaluate(
                        action=action,
                        image=str(image_path),
                        use_few_shot=use_few_shot,
                        use_thinking=use_thinking
                    )
                    
                    # Save result
                    result = {
                        "idx": item["idx"],
                        "type": item["type"],
                        "scenario_type": item["scenario_type"],
                        "csv_row": row_dict,
                        "action": action,
                        "image_path": str(image_path),
                        "messages": messages_for_storage,  # Save messages sent to model (with image_path instead of base64)
                        "response": evaluation_result["response"],
                        "parsed_response": evaluation_result["parsed_response"],
                        "usage": evaluation_result["usage"],
                        "cost": evaluation_result["cost"],
                    }
                    
                    # Include ID if present in CSV
                    row_id = row_dict.get("id") or row_dict.get("ID")
                    if row_id:
                        result["id"] = str(row_id)
                    
                    results.append(result)
                    total_cost += evaluation_result["cost"]
                    total_usage["prompt_tokens"] += evaluation_result["usage"].get("prompt_tokens", 0)
                    total_usage["completion_tokens"] += evaluation_result["usage"].get("completion_tokens", 0)
                    total_usage["total_tokens"] += evaluation_result["usage"].get("total_tokens", 0)
                    
                    # Save intermediate results
                    if save_intermediate and len(results) % 10 == 0:
                        self._save_results(results, output_path)
                    
                except Exception as e:
                    print(f"Error processing row {item['idx']}: {e}")
                    error_result = {
                        "idx": item["idx"],
                        "type": item["type"],
                        "scenario_type": item["scenario_type"],
                        "csv_row": item["row"],
                        "error": str(e),
                    }
                    results.append(error_result)
        else:
            # Parallel inference
            print(f"Starting parallel inference on {len(dataset)} items with {num_workers} workers...")
            # Add split-level safety type for legacy parallel callers.
            worker_config["dataset_type"] = dataset_type
            results, total_cost, total_usage = run_parallel_inference(
                dataset=dataset,
                config=worker_config,
                num_workers=num_workers,
                description="Running inference"
            )
            
            # Handle errors in results
            for result in results:
                if result is None:
                    continue
                if "error" not in result:
                    # Save intermediate results
                    if save_intermediate and len([r for r in results if r and "error" not in r]) % 10 == 0:
                        self._save_results([r for r in results if r], output_path)
        
        # Save final results
        self._save_results(results, output_path)
        
        # Print statistics
        print(f"\n=== Evaluation Complete ===")
        print(f"Dataset type: {dataset_type}")
        print(f"Total items processed: {len(results)}")
        print(f"Success: {len([r for r in results if r and 'error' not in r])}")
        print(f"Failed: {len([r for r in results if r and 'error' in r])}")
        print(f"Total cost: ${total_cost:.4f}")
        print(f"Total token usage:")
        print(f"  - Prompt: {total_usage['prompt_tokens']:,}")
        print(f"  - Completion: {total_usage['completion_tokens']:,}")
        print(f"  - Total: {total_usage['total_tokens']:,}")
        print(f"Results saved to: {output_path}")
        
        return results
    
    
    def _save_results(self, results: List[Dict[str, Any]], output_path: Path):
        """Save results in JSONL format"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False, default=str) + '\n')


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
        split: Split name for Hugging Face dataset (e.g., "safe", "unsafe")
        **kwargs: Additional model settings (temperature, max_tokens, use_thinking, etc.)
        
    Returns:
        List of evaluation results
    """
    # Load config
    if config_path:
        from utils.config import load_config
        config = load_config(config_path)
    else:
        config = get_config()
    
    # Get use_thinking from config if not provided in kwargs
    common_config = config.get("common", {})
    default_use_thinking = common_config.get("use_thinking", False)
    
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
    
    # Handle vLLM port override (modify base_url port if vllm_port is provided)
    if provider.lower() == "vllm" and "vllm_port" in kwargs and kwargs["vllm_port"]:
        port = kwargs["vllm_port"]
        # Get base_url from config or use default
        base_url = model_config.get("base_url", "http://127.0.0.1:8000/v1")
        
        # Replace port in base_url using regex
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
    
    # Additional settings (kwargs takes precedence - full base_url override)
    if "base_url" in kwargs and kwargs["base_url"]:
        model_config["base_url"] = kwargs["base_url"]
    
    # Run evaluation
    evaluator = EMBGuardHeldout(provider, model_config, kwargs.get("output_dir"))
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
        Dictionary mapping dataset names (safe, unsafe) to file paths
    """
    # Load config
    if config_path:
        from utils.config import load_config
        config = load_config(config_path)
    else:
        config = get_config()
    
    # Get heldout_set paths from common section
    common_config = config.get("common", {})
    heldout_set_config = common_config.get("heldout_set", {})
    if not heldout_set_config:
        # Fallback: use default paths
        project_path = get_project_path()
        return {
            "safe": str(project_path / "data" / "heldout_set" / "dataset_safe.csv"),
            "unsafe": str(project_path / "data" / "heldout_set" / "dataset_unsafe.csv"),
        }
    
    # Resolve relative paths to absolute paths
    project_path = get_project_path()
    heldout_set_paths = {}
    for key in ["safe", "unsafe"]:
        if key in heldout_set_config:
            path = heldout_set_config[key]
            # Convert relative path to absolute
            if not Path(path).is_absolute():
                path = project_path / path
            else:
                path = Path(path)
            heldout_set_paths[key] = str(path)
        else:
            # Fallback: use default path
            default_path = project_path / "data" / "heldout_set" / f"dataset_{key}.csv"
            heldout_set_paths[key] = str(default_path)
    
    return heldout_set_paths


def evaluate_heldout_sets(
    provider: str,
    model_name: str,
    datasets: List[str],
    config_path: Optional[str] = None,
    **kwargs
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Evaluate specified heldout datasets
    
    Args:
        provider: LLM provider
        model_name: Model name
        datasets: List of dataset names to evaluate (e.g., ["safe", "unsafe"])
        config_path: Config file path (uses default if None)
        **kwargs: Additional model settings
        
    Returns:
        Dictionary mapping dataset names to results
    """
    # Get heldout set paths from config or defaults
    heldout_set_paths = get_heldout_set_paths(config_path)
    
    # Validate dataset names
    valid_datasets = ["safe", "unsafe"]
    invalid_datasets = [ds for ds in datasets if ds.lower() not in valid_datasets]
    if invalid_datasets:
        raise ValueError(f"Invalid dataset names: {invalid_datasets}. Valid names: {valid_datasets}")
    
    all_results = {}
    for dataset in datasets:
        dataset_lower = dataset.lower()
        if dataset_lower not in heldout_set_paths:
            print(f"Warning: Dataset '{dataset_lower}' not found, skipping...")
            continue
        
        csv_path = heldout_set_paths[dataset_lower]
        print(f"\n{'='*60}")
        print(f"Evaluating: {dataset_lower.upper()} ({csv_path})")
        print(f"{'='*60}")
        
        try:
            results = evaluate_from_config(
                provider=provider,
                model_name=model_name,
                csv_path=csv_path,
                config_path=config_path,
                **kwargs
            )
            all_results[dataset_lower] = results
        except Exception as e:
            print(f"Error evaluating {dataset_lower}: {e}")
            all_results[dataset_lower] = []
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models on heldout set CSV files")
    parser.add_argument("--provider", type=str, required=True,
                       choices=["openai", "openrouter", "vllm", "claude", "gemini"],
                       help="LLM provider")
    parser.add_argument("--model", type=str, required=True,
                       help="Model name")
    
    # Dataset selection arguments
    parser.add_argument("--dataset", type=str, default="all",
                       help="Datasets to evaluate: 'all', 'safe', 'unsafe', or comma-separated (e.g., 'safe,unsafe'). Default: 'all'")
    parser.add_argument("--csv", type=str, default=None,
                       help="Path to specific CSV file (overrides config and --dataset)")
    
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature setting")
    parser.add_argument("--max_tokens", type=int, default=2048,
                       help="Maximum number of tokens")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory")
    parser.add_argument("--no-few-shot", action="store_true",
                       help="Disable few-shot examples in prompts")
    parser.add_argument("--use-thinking", action="store_true",
                       help="Enable thinking mode (step-by-step reasoning) for VLM evaluation")
    parser.add_argument("--num-workers", type=int, default=1,
                       help="Number of worker processes for parallel evaluation (default: 1, sequential)")
    parser.add_argument("--base-url", type=str, default=None,
                       help="Base URL for API (overrides config.yaml, useful for vLLM with custom port, e.g., http://127.0.0.1:8000/v1)")
    parser.add_argument("--vllm-port", type=str, default=None,
                       help="Port number for vLLM (overrides port in config.yaml base_url, e.g., 8000, 8008)")
    
    args = parser.parse_args()
    
    # Determine which datasets to evaluate
    if args.data_source:
        # Evaluate single CSV file (custom path)
        results = evaluate_from_config(
            provider=args.provider,
            model_name=args.model,
            data_source=args.data_source,
            split=args.split,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            output_dir=args.output_dir,
            use_few_shot=not args.no_few_shot,
            use_thinking=args.use_thinking,
            num_workers=args.num_workers,
            base_url=args.base_url,
            vllm_port=args.vllm_port,
        )
        print(f"\nGenerated {len(results)} results")
    else:
        # Parse dataset argument
        dataset_arg = args.dataset.lower().strip()
        
        if dataset_arg == "all":
            datasets = ["safe", "unsafe"]
        else:
            # Parse comma-separated values
            datasets = [ds.strip().lower() for ds in dataset_arg.split(",")]
            # Validate dataset names
            valid_datasets = ["safe", "unsafe"]
            invalid_datasets = [ds for ds in datasets if ds not in valid_datasets]
            if invalid_datasets:
                raise ValueError(f"Invalid dataset names: {invalid_datasets}. Valid names: {valid_datasets} or 'all'")
        
        # Evaluate selected datasets
        all_results = evaluate_heldout_sets(
            provider=args.provider,
            model_name=args.model,
            datasets=datasets,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            output_dir=args.output_dir,
            use_few_shot=not args.no_few_shot,
            use_thinking=args.use_thinking,
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
