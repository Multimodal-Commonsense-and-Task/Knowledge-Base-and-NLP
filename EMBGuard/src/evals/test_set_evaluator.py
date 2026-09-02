"""
Test set evaluator for EMBGuard
Evaluates models on test set CSV files or Hugging Face datasets
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import pandas as pd

from utils.config import get_config
from utils.path import get_project_path
from utils.test_types import (
    extract_test_type_code_from_text,
    normalize_test_type_code,
    test_type_safety_type,
    test_type_slug,
)
from src.guardrail.guardrail import EMBGuard
from src.inference_utils import run_parallel_inference
from src.evals.utils import load_data, resolve_image, convert_messages_for_storage


class TestSetEvaluator:
    """Evaluator for test set CSV files or Hugging Face datasets"""
    
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
            output_dir: Output directory for results (auto-generated if None)
        """
        self.provider = provider
        self.model_config = model_config
        self.guard = EMBGuard(provider, model_config)
        # Note: output_dir parameter is kept for backward compatibility but not used in run() method
        # run() method uses outputs/EMBGuardTest/{provider}_{model_name} instead
        self.output_dir = output_dir
    
    def run(
        self,
        data_source: str,
        output_filename: Optional[str] = None,
        save_intermediate: bool = True,
        use_few_shot: bool = True,
        use_thinking: bool = False,
        num_workers: int = 1,
        test_set_type: Optional[str] = None,
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
            test_set_type: Test set label or alias (for example, "causal_risky"
                or "Causal Risky"). Auto-detected from filename if None.
            split: Split name for Hugging Face dataset. Accepts legacy codes
                (e.g., "HR") and full split aliases (e.g., "causal_risky").
            
        Returns:
            List of evaluation results
        """
        split = normalize_test_type_code(split, default=split)
        test_set_type = normalize_test_type_code(test_set_type, default=test_set_type)

        # Load data (CSV or Hugging Face dataset)
        df, is_hf_dataset, csv_dir = load_data(data_source, split=split)
        
        # Extract test set type from filename or split if not provided
        if test_set_type is None:
            if is_hf_dataset:
                # For HF datasets, use split name if provided
                if split:
                    test_set_type = normalize_test_type_code(split, default=split)
                else:
                    # Try row-level normalized schema first, then legacy Type.
                    for type_column in ("scenario_type", "Type", "Subtype"):
                        if type_column in df.columns and len(df) > 0:
                            test_set_type = normalize_test_type_code(df.iloc[0].get(type_column, ""), default="UNKNOWN")
                            break
                    if not test_set_type:
                        test_set_type = "UNKNOWN"
            else:
                # For CSV files, extract from filename
                csv_path_obj = Path(data_source)
                csv_name = csv_path_obj.stem.upper()  # e.g., "test_dataset_HR"
                # Try to extract an EMBGuardTest split name or legacy code from filename.
                test_set_type = extract_test_type_code_from_text(csv_name)
                if test_set_type is None:
                    # Fallback: try to get from row-level normalized or legacy type columns.
                    for type_column in ("scenario_type", "Type", "Subtype"):
                        if type_column in df.columns and len(df) > 0:
                            test_set_type = normalize_test_type_code(df.iloc[0].get(type_column, ""), default="UNKNOWN")
                            break
                    if not test_set_type:
                        test_set_type = "UNKNOWN"
        
        # Set output directory
        # Use custom output_dir if provided, otherwise use default: outputs/EMBGuardTest/{provider}_{model_name}
        if self.output_dir:
            dataset_output_dir = Path(self.output_dir)
        else:
            project_path = get_project_path()
            model_name = self.model_config.get("model_name", "model")
            model_name_clean = model_name.replace("/", "_").replace("\\", "_")
            provider_model = f"{self.provider}_{model_name_clean}"
            
            # Ensure we create the full path correctly: outputs/EMBGuardTest/{provider}_{model_name}
            base_output_dir = project_path / "outputs" / "EMBGuardTest"
            base_output_dir.mkdir(parents=True, exist_ok=True)
            dataset_output_dir = base_output_dir / provider_model
        
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get model name for filename generation
        model_name = self.model_config.get("model_name", "model")
        model_name_clean = model_name.replace("/", "_").replace("\\", "_")
        provider_model = f"{self.provider}_{model_name_clean}"
        
        # Set output file path
        if output_filename is None:
            if is_hf_dataset:
                # For HF datasets, use dataset name and split
                dataset_name = data_source.replace("/", "_")
                split_name = split or test_set_type or "unknown"
                csv_name = f"{dataset_name}_{split_name}"
            else:
                csv_name = Path(data_source).stem
            
            # Build condition suffix
            few_shot_suffix = "few-shot" if use_few_shot else "no-few-shot"
            thinking_suffix = "thinking" if use_thinking else "non-thinking"
            condition_suffix = f"{few_shot_suffix}_{thinking_suffix}"
            
            output_filename = f"{provider_model}_{csv_name}_{condition_suffix}_results.jsonl"
        
        output_path = dataset_output_dir / output_filename
        
        # Prepare dataset for evaluation
        dataset = []
        for idx, row in df.iterrows():
            row_dict = row.to_dict() if hasattr(row, 'to_dict') else row
            
            # Check for image (HF dataset has "image"; CSV/HF fallback has image_path).
            has_image = False
            image_path_value = ""
            
            if is_hf_dataset:
                # For HF datasets, check for "image" column (PIL Image) or path fallback.
                if "image" in row_dict and row_dict["image"] is not None:
                    has_image = True
                else:
                    image_path_value = (
                        row_dict.get("image_path", "")
                        or row_dict.get("source_path", "")
                        or row_dict.get("URL", "")
                        or row_dict.get("url", "")
                    )
                if image_path_value:
                    has_image = True
            else:
                # For CSV files, check for normalized image_path or legacy URL/path.
                image_path_value = (
                    row_dict.get("image_path", "")
                    or row_dict.get("source_path", "")
                    or row_dict.get("URL", "")
                    or row_dict.get("url", "")
                    or row_dict.get("path", "")
                )
                if image_path_value and not (pd.isna(image_path_value) or image_path_value == ""):
                    has_image = True
            
            if not has_image:
                continue  # Skip rows without image
            
            dataset.append({
                "idx": int(idx),
                "row": row_dict,
                "image_path": image_path_value,
                "csv_dir": str(csv_dir) if csv_dir else None,
                "is_hf_dataset": is_hf_dataset,
            })
        
        # Prepare config for workers
        worker_config = {
            "provider": self.provider,
            "model_config": self.model_config,
            "use_few_shot": use_few_shot,
            "use_thinking": use_thinking,
            "test_set_type": test_set_type,
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
            
            for item in tqdm(dataset, desc="Evaluating"):
                try:
                    # Extract action and image from row
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
                    safety_type = str(row_dict.get("type") or "").strip().lower()
                    if safety_type not in {"safe", "unsafe"}:
                        safety_type = test_type_safety_type(test_set_type) or "unknown"
                    result = {
                        "idx": item["idx"],
                        "type": safety_type,
                        "scenario_type": test_type_slug(test_set_type),
                        "csv_row": row_dict,
                        "action": action,
                        "image_path": str(image_path),
                        "messages": messages_for_storage,
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
                    safety_type = test_type_safety_type(test_set_type) or "unknown"
                    error_result = {
                        "idx": item["idx"],
                        "type": safety_type,
                        "scenario_type": test_type_slug(test_set_type),
                        "csv_row": item["row"],
                        "error": str(e),
                    }
                    results.append(error_result)
        else:
            # Parallel inference
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
