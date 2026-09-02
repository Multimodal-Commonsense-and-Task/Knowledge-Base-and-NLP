"""
EMBGuard Train Data Construction Task
Converts dataset CSV or Hugging Face dataset to OpenAI format for multimodal SFT training.
"""
from typing import List, Dict, Any, Optional
import json
import os
import csv
from pathlib import Path
from tqdm import tqdm

from src.dataset_generation.task import BaseTask
from src.guardrail.prompts.guardrail_prompt import (
    GUARDRAIL_SYSTEM_PROMPT,
    GUARDRAIL_USER_PROMPT,
    get_few_shot_messages
)
from utils.test_types import hf_split_candidates, normalize_test_type_code


class EMBGuardTrainDataConstructionTask(BaseTask):
    """Task: Construct training data for EMBGuard in OpenAI format"""
    
    def get_task_name(self) -> str:
        return "embguard_train_data_construction"
    
    def _is_huggingface_dataset(self, path: Path) -> bool:
        """
        Check if the given path is a Hugging Face dataset directory.
        
        A Hugging Face dataset directory typically contains:
        - A subdirectory with split name (e.g., 'train', 'validation')
        - Or dataset_info.json file
        """
        if not path.is_dir():
            return False
        
        # Check for common split directories
        common_splits = ['train', 'validation', 'test', 'dev']
        for split in common_splits:
            split_dir = path / split
            if split_dir.exists() and split_dir.is_dir():
                # Check if it looks like a HF dataset (has state.json or arrow files)
                if (split_dir / 'state.json').exists() or any(
                    f.suffix == '.arrow' for f in split_dir.iterdir()
                ):
                    return True
        
        # Check for dataset_info.json at root
        if (path / 'dataset_info.json').exists():
            return True
        
        return False
    
    def _load_huggingface_dataset(self, dataset_path: Path, split: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load data from Hugging Face dataset directory or Hub.
        
        Args:
            dataset_path: Path to Hugging Face dataset directory or dataset ID
            split: Dataset split to load (e.g., 'train'). If None, tries to auto-detect.
        
        Returns:
            List of dictionaries containing dataset rows
        """
        try:
            from datasets import load_from_disk, load_dataset
        except ImportError:
            raise ImportError(
                "datasets library is required to load Hugging Face datasets. "
                "Install it via: pip install datasets"
            )
        
        # Check if dataset_path is a directory (saved dataset) or a string (dataset ID)
        dataset_path_str = str(dataset_path)
        split = normalize_test_type_code(split, default=split)
        
        # Try to load from disk first (if it's a saved dataset)
        if Path(dataset_path).is_dir():
            # Auto-detect split if not provided
            if split is None:
                common_splits = ['train', 'validation', 'test', 'dev']
                for s in common_splits:
                    split_dir = dataset_path / s
                    if split_dir.exists() and split_dir.is_dir():
                        split = s
                        break
                
                if split is None:
                    # Try loading directly from the directory (might be a single split)
                    print(f"Loading Hugging Face dataset from: {dataset_path}")
                    try:
                        dataset = load_from_disk(str(dataset_path))
                        split = None  # Single dataset, no split
                    except Exception:
                        raise ValueError(
                            f"Could not auto-detect split in dataset directory: {dataset_path}. "
                            f"Please specify split name or ensure one of {common_splits} exists."
                        )
            
            if split is not None:
                split_path = dataset_path / split
                if not split_path.exists():
                    raise FileNotFoundError(
                        f"Dataset split '{split}' not found in {dataset_path}. "
                        f"Available splits: {[d.name for d in dataset_path.iterdir() if d.is_dir()]}"
                    )
                print(f"Loading Hugging Face dataset from: {split_path}")
                dataset = load_from_disk(str(split_path))
            else:
                print(f"Loading Hugging Face dataset from: {dataset_path}")
                dataset = load_from_disk(str(dataset_path))
        else:
            # It's a dataset ID, load from Hub (will use cache automatically)
            if split is None:
                split = 'train'  # Default to train split
            split = normalize_test_type_code(split, default=split)
            
            print(f"Loading Hugging Face dataset from Hub: {dataset_path_str} (split: {split})")
            # Use cache directory from environment if available
            # HF_DATASETS_CACHE takes precedence over HF_HOME
            cache_dir = os.getenv('HF_DATASETS_CACHE')
            if not cache_dir:
                hf_home = os.getenv('HF_HOME')
                if hf_home:
                    cache_dir = os.path.join(hf_home, 'datasets')
            
            if cache_dir:
                print(f"Using cache directory: {cache_dir}")
                last_error = None
                for split_candidate in hf_split_candidates(split):
                    try:
                        dataset = load_dataset(dataset_path_str, split=split_candidate, cache_dir=cache_dir)
                        split = split_candidate
                        break
                    except Exception as e:
                        last_error = e
                else:
                    raise last_error
            else:
                # Will use default cache location
                last_error = None
                for split_candidate in hf_split_candidates(split):
                    try:
                        dataset = load_dataset(dataset_path_str, split=split_candidate)
                        split = split_candidate
                        break
                    except Exception as e:
                        last_error = e
                else:
                    raise last_error
        
        # Print dataset info for debugging
        if len(dataset) > 0:
            print(f"Dataset features: {list(dataset[0].keys())}")
            print(f"Sample Action: {dataset[0].get('Action', 'N/A')}")
            print(f"Sample URL: {dataset[0].get('URL', 'N/A')}")
        
        # Convert to list of dictionaries with progress tracking
        data = []
        total_items = len(dataset)
        skipped_count = 0
        
        for item in tqdm(dataset, desc="Loading dataset", total=total_items, unit="examples"):
            # Convert to dict, handling Image objects
            row = {}
            for key, value in item.items():
                if key == 'image':
                    # Skip image column - we'll use URL column for image paths
                    continue
                # Convert value to string if needed, handle None
                if value is None:
                    row[key] = ''
                else:
                    # Preserve original type for non-None values, but convert to string for consistency
                    row[key] = str(value) if not isinstance(value, (int, float, bool)) else value
            
            # Filter out rows without action or URL (case-insensitive check)
            action = row.get('Action') or row.get('action') or ''
            url = row.get('URL') or row.get('url') or ''
            
            if not action or not url:
                skipped_count += 1
                continue
            
            # Normalize keys to have 'Action' and 'URL' (capitalized)
            if 'action' in row and 'Action' not in row:
                row['Action'] = row.pop('action')
            if 'url' in row and 'URL' not in row:
                row['URL'] = row.pop('url')
            
            data.append(row)
        
        print(f"Loaded {len(data)} examples from Hugging Face dataset")
        if skipped_count > 0:
            print(f"  Skipped {skipped_count} examples (missing Action or URL)")
        return data
    
    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load data from CSV file or Hugging Face dataset.
        
        Supports:
        - CSV file: Path to a CSV file
        - Hugging Face dataset: Path to a directory containing HF dataset
        
        Expected columns:
        - Action: The action to be evaluated
        - URL: Path to the image file
        - Situation: Scene description (optional, for metadata)
        - Category, Subcategory, Type, etc.: Metadata fields
        """
        if not self.args.data_path:
            raise ValueError(
                "data_path is required for embguard_train_data_construction task. "
                "Please provide --data_path with CSV file path or Hugging Face dataset directory."
            )
        
        data_path_str = self.args.data_path
        
        # Check if it's a dataset ID (contains '/') or a path
        if '/' in data_path_str and not Path(data_path_str).exists():
            # Likely a dataset ID (e.g., "EMBGuard/EMBHazard_original_wo_filter_v1.0")
            # Check if it looks like a dataset ID (has organization/dataset format)
            if data_path_str.count('/') == 1 and not data_path_str.startswith('/'):
                print(f"Detected dataset ID: {data_path_str}")
                # Get split from args if available
                split = getattr(self.args, 'dataset_split', None)
                return self._load_huggingface_dataset(Path(data_path_str), split=split)
        
        data_path = Path(data_path_str)
        if not data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")
        
        # Check if it's a Hugging Face dataset directory
        if self._is_huggingface_dataset(data_path):
            # Auto-detect split (task will find train/validation/test automatically)
            split = getattr(self.args, 'dataset_split', None)
            return self._load_huggingface_dataset(data_path, split=split)
        
        # Otherwise, treat as CSV file
        if not data_path.is_file():
            raise ValueError(
                f"Path {data_path} is neither a CSV file nor a Hugging Face dataset directory."
            )
        
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Filter out rows without action or URL
                if not row.get('Action') or not row.get('URL'):
                    continue
                
                data.append(row)
        
        return data
    
    def create_prompter(self):
        """Not needed for this task - we construct messages directly"""
        return None
    
    def prepare_model_inputs(self, dataset: List[Dict[str, Any]], prompter) -> List:
        """
        Prepare training data in OpenAI format.
        This task doesn't need inference, so we directly construct the training data.
        """
        # For this task, we don't need to prepare model inputs for inference
        # Instead, we'll construct training data directly in gather_results
        return []
    
    def process_result(self, response: Dict[str, Any], model_input: List) -> Dict[str, Any]:
        """
        Not used for this task - we're constructing training data, not running inference.
        """
        return {}
    
    def gather_results(self, results_file: str, scenarios_file: str, output_file: str, **kwargs) -> Dict[str, Any]:
        """
        Construct training data in OpenAI format for Hugging Face upload.
        
        This task constructs training data directly from loaded dataset (CSV or Hugging Face).
        
        Args:
            results_file: Not used (we construct from data directly)
            scenarios_file: Not used (we use data_path from args)
            output_file: Output JSON file path
            **kwargs: Additional arguments
        """
        # Use the loaded dataset (from load_data)
        # We need to reload it here since gather_results is called separately
        dataset = self.load_data()
        
        # Get project root for path resolution
        project_root = Path(__file__).parents[3]
        
        # Get CSV directory for resolving relative image paths
        # If data_path is a CSV file, use its directory as base for images
        csv_dir = None
        if self.args.data_path:
            data_path = Path(self.args.data_path)
            if data_path.is_file() and data_path.suffix == '.csv':
                csv_dir = data_path.parent
        
        # Construct training data with progress tracking
        training_data = []
        use_few_shot = getattr(self.args, 'use_few_shot', True)
        
        skipped_no_action = 0
        skipped_no_url = 0
        skipped_no_image = 0
        
        for row in tqdm(dataset, desc="Constructing training data", unit="examples"):
            action = row.get('Action', '').strip()
            image_path_str = row.get('URL', '').strip()

            if not action:
                skipped_no_action += 1
                continue

            if not image_path_str:
                skipped_no_url += 1
                continue

            # Resolve image path
            # Try CSV directory first (for relative paths like "images/xxx.jpg")
            # Then try project root, then absolute path
            if not os.path.isabs(image_path_str):
                # Try relative to CSV directory first
                if csv_dir:
                    image_path = csv_dir / image_path_str
                    if not image_path.exists():
                        # Fallback to project root
                        image_path = project_root / image_path_str
                else:
                    # No CSV directory, use project root
                    image_path = project_root / image_path_str
            else:
                image_path = Path(image_path_str)

            # Skip if image doesn't exist
            if not image_path.exists():
                skipped_no_image += 1
                if skipped_no_image <= 5:  # Print first few missing images for debugging
                    print(f"Warning: Image not found: {image_path} (from URL: {image_path_str})")
                continue

            # Convert to relative path for Hugging Face (relative to project root)
            try:
                image_rel_path = os.path.relpath(image_path, project_root)
                if image_rel_path.startswith(".."):
                    # If outside project root, keep absolute
                    image_rel_path = str(image_path)
            except ValueError:
                image_rel_path = str(image_path)

            # Construct messages in OpenAI format
            messages = []

            # System message
            messages.append({
                "role": "system",
                "content": GUARDRAIL_SYSTEM_PROMPT
            })

            # Add few-shot examples if enabled
            if use_few_shot:
                few_shot_messages = get_few_shot_messages()
                # Add <image> token to few-shot user messages if they should have images
                # Note: Few-shot examples don't have actual images, but we add the token for consistency
                for msg in few_shot_messages:
                    if msg.get("role") == "user" and "<image>" not in msg.get("content", ""):
                        msg["content"] = f"<image>{msg['content']}"
                messages.extend(few_shot_messages)

            # User message with action and image
            # Add <image> token at the beginning of the prompt for multimodal models
            # LlamaFactory requires: number of <image> tokens == length of images array
            user_prompt = GUARDRAIL_USER_PROMPT.format(action=action)
            user_message = {
                "role": "user",
                "content": f"<image>{user_prompt}",  # Add <image> token for LlamaFactory
                "images": [image_rel_path]  # Add image to user message
            }
            messages.append(user_message)

            # Assistant message (expected response)
            # Construct expected JSON response from data
            risk = row.get('Risk', '').strip()
            risk_type = row.get('Risk Type', '').strip()
            hazard = row.get('Related Hazard', '').strip()

            potential_risk = "unsafe" if risk == "O" else "safe"
            if potential_risk == "safe":
                risk_type = "none"
                hazard = "none"

            assistant_response = json.dumps({
                "potential_risk": potential_risk,
                "risk_type": risk_type if risk_type else "none",
                "hazard": hazard if hazard else "none"
            }, ensure_ascii=False)

            messages.append({
                "role": "assistant",
                "content": assistant_response
            })

            # Create training example in OpenAI format
            training_example = {
                "messages": messages,
                "images": [image_rel_path]  # Top-level images array for LlamaFactory
            }

            training_data.append(training_example)
        
        # Save to output file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        result = {
            "metadata": {
                "total_examples": len(training_data),
                "format": "openai",
                "task": "embguard_train_data_construction",
                "use_few_shot": use_few_shot
            },
            "data": training_data
        }
        
        print(f"✓ Created {len(training_data)} training examples")
        print(f"  Output: {output_file}")
        
        # Print statistics
        if skipped_no_action > 0:
            print(f"  Skipped {skipped_no_action} examples (no Action)")
        if skipped_no_url > 0:
            print(f"  Skipped {skipped_no_url} examples (no URL)")
        if skipped_no_image > 0:
            print(f"  Skipped {skipped_no_image} examples (image file not found)")
        
        if len(training_data) == 0:
            print(f"\n⚠ Warning: No training examples were created!")
            print(f"  Total dataset size: {len(dataset)}")
            print(f"  Check if:")
            print(f"    1. Dataset has 'Action' and 'URL' columns")
            print(f"    2. Image files exist at the paths specified in 'URL' column")
            if len(dataset) > 0:
                print(f"  First row sample: {list(dataset[0].keys())}")
                print(f"  First row Action: {dataset[0].get('Action', 'N/A')}")
                print(f"  First row URL: {dataset[0].get('URL', 'N/A')}")
        
        return result
