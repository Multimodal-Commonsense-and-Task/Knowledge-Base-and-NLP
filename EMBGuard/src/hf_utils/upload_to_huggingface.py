#!/usr/bin/env python3
"""
Upload EMBGuardTest and heldout_set datasets to Hugging Face.

Usage:
    python -m src.hf_utils.upload_to_huggingface --org <organization_name> [--dataset <dataset_name>]
    
Example:
    python -m src.hf_utils.upload_to_huggingface --org your-org --dataset EMBGuardTest
    python -m src.hf_utils.upload_to_huggingface --org your-org --dataset heldout_set
    python -m src.hf_utils.upload_to_huggingface --org your-org  # Uploads both datasets
"""

import argparse
import os
import pandas as pd
from pathlib import Path
from PIL import Image as PILImage
from datasets import Dataset, DatasetDict, Image, Features, Value
from huggingface_hub import HfApi, login
from tqdm import tqdm
import sys


def get_project_root():
    """Get the project root directory."""
    # This file is in src/hf_utils/, so go up two levels to get project root
    script_dir = Path(__file__).parent
    return script_dir.parent.parent


project_root = get_project_root()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.test_types import TEST_TYPE_CODES, normalize_test_type_code, test_type_label, test_type_slug
from src.hf_utils.migrate_hf_datasets import _normalize_public_schema


def load_csv_with_images(csv_path, base_dir, image_column="URL"):
    """
    Load CSV file and include images as Image objects.
    
    Args:
        csv_path: Path to CSV file
        base_dir: Base directory for resolving image paths
        image_column: Name of the column containing image paths
        
    Returns:
        List of dictionaries with data and images
    """
    df = pd.read_csv(csv_path)
    if image_column not in df.columns:
        for candidate in ("image_path", "source_path", "URL", "url", "path"):
            if candidate in df.columns:
                image_column = candidate
                break
    base_dir = Path(base_dir)
    
    data = []
    missing_images = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Loading images"):
        item = row.to_dict()
        # Ensure optional columns exist for consistent schema
        if "Pair Item ID" not in item:
            item["Pair Item ID"] = ""
        if "Room" not in item:
            item["Room"] = ""
        
        # Convert all values to native Python types (handle NaN, etc.)
        # Convert NaN/None to empty string for string columns to ensure consistent schema
        for key, value in item.items():
            if pd.isna(value) or value == "":
                item[key] = ""  # Use empty string instead of None for consistency
            elif isinstance(value, (pd.Int64Dtype, pd.Float64Dtype)):
                item[key] = value.item() if hasattr(value, 'item') else value
            else:
                # Ensure all values are strings
                item[key] = str(value) if value is not None else ""
        
        # Handle image path - load as PIL Image
        if image_column in item and item[image_column] is not None and item[image_column] != "":
            image_path_str = str(item[image_column])
            # Check if path is absolute or relative
            if os.path.isabs(image_path_str):
                image_path = Path(image_path_str)
            else:
                # Relative path: resolve from base_dir
                image_path = base_dir / image_path_str
            if image_path.exists() and image_path.is_file():
                try:
                    # Load image as PIL Image for Hugging Face datasets
                    # Use path directly to avoid file handle issues
                    pil_image = PILImage.open(image_path)
                    # Convert to RGB if needed and load immediately
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert("RGB")
                    # Load the image data immediately to avoid lazy loading issues
                    pil_image.load()
                    item["image"] = pil_image
                except Exception as e:
                    print(f"Warning: Failed to load image {image_path}: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    item["image"] = None
                    missing_images += 1
            else:
                if not image_path.exists():
                    print(f"Warning: Image not found: {image_path}")
                item["image"] = None
                missing_images += 1
        else:
            item["image"] = None
        
        data.append(item)
    
    if missing_images > 0:
        print(f"  Warning: {missing_images} images could not be loaded")
    
    return data


def create_embguardtest_dataset(data_dir, test_split_names="full", schema="v3"):
    """Create EMBGuardTest dataset from CSV files."""
    data_dir = Path(data_dir)
    
    datasets = {}
    
    # Define consistent features schema for all splits
    # All text columns should be strings, image can be null
    features = Features({
        "Category": Value("string"),
        "Subcategory": Value("string"),
        "Type": Value("string"),
        "Type Label": Value("string"),
        "ID": Value("string"),
        "Situation": Value("string"),
        "Action": Value("string"),
        "Risk": Value("string"),
        "Risk Type": Value("string"),
        "Related Hazard": Value("string"),
        "Pair Item ID": Value("string"),
        "Room": Value("string"),
        "URL": Value("string"),
        "image": Image(decode=True),
    })
    
    # Load each test dataset split
    splits = list(TEST_TYPE_CODES)
    for split in splits:
        csv_file = data_dir / f"test_dataset_{split}.csv"
        if csv_file.exists():
            split_code = normalize_test_type_code(split, strict=True)
            split_name = test_type_slug(split_code) if test_split_names == "full" else split_code
            print(f"Loading {split_code} split as {split_name}...")
            data = load_csv_with_images(csv_file, data_dir)
            for item in data:
                item["Type"] = split_code
                item["Type Label"] = test_type_label(split_code)
            dataset = Dataset.from_list(data, features=features)
            if schema == "v3":
                dataset = _normalize_public_schema(
                    dataset,
                    split_name=split_name,
                    force_scenario_code=split_code,
                )
            datasets[split_name] = dataset
            print(f"  Loaded {len(data)} examples")
        else:
            print(f"Warning: {csv_file} not found, skipping {split} split")
    
    if not datasets:
        raise ValueError(f"No valid splits found in {data_dir}")
    
    return DatasetDict(datasets)


def create_heldout_dataset(data_dir, schema="v3"):
    """Create heldout_set dataset from CSV files."""
    data_dir = Path(data_dir)
    
    datasets = {}
    
    # Define consistent features schema for all splits
    features = Features({
        "Category": Value("string"),
        "Subcategory": Value("string"),
        "Type": Value("string"),
        "ID": Value("string"),
        "Situation": Value("string"),
        "Action": Value("string"),
        "Risk": Value("string"),
        "Risk Type": Value("string"),
        "Related Hazard": Value("string"),
        "Pair Item ID": Value("string"),
        "Room": Value("string"),
        "Mitigate Action": Value("string"),
        "URL": Value("string"),
        "image": Image(decode=True),
    })
    
    # Load safe and unsafe datasets
    for split_name, csv_file in [("safe", "dataset_safe.csv"), ("unsafe", "dataset_unsafe.csv")]:
        csv_path = data_dir / csv_file
        if csv_path.exists():
            print(f"Loading {split_name} split...")
            data = load_csv_with_images(csv_path, data_dir)
            dataset = Dataset.from_list(data, features=features)
            if schema == "v3":
                dataset = _normalize_public_schema(dataset, split_name=split_name)
            datasets[split_name] = dataset
            print(f"  Loaded {len(data)} examples")
        else:
            print(f"Warning: {csv_path} not found, skipping {split_name} split")
    
    if not datasets:
        raise ValueError(f"No valid splits found in {data_dir}")
    
    return DatasetDict(datasets)


def load_json_with_images(json_path, base_dir):
    """
    Load JSON file (OpenAI format training data) and include images as Image objects.
    
    Args:
        json_path: Path to JSON file
        base_dir: Base directory for resolving image paths
        
    Returns:
        List of dictionaries with messages and images (PIL Images)
    """
    import json
    
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    base_dir = Path(base_dir)
    
    print(f"  Reading JSON file...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"JSON file must contain a list of examples, got {type(data)}")
    
    print(f"  Found {len(data)} items in JSON file")
    print(f"  Processing items and loading images...")
    
    processed_data = []
    missing_images = 0
    
    # Use tqdm for progress bar
    for idx, item in tqdm(enumerate(data), total=len(data), desc="  Processing items"):
        if not isinstance(item, dict):
            print(f"Warning: Skipping item {idx} - not a dictionary")
            continue
        
        # Extract messages and images
        messages = item.get("messages", [])
        top_level_images = item.get("images", [])
        
        if not messages:
            print(f"Warning: Skipping item {idx} - no messages field")
            continue
        
        # Process top-level images - store both PIL Image and original path
        processed_top_images = []
        processed_top_image_paths = []  # Store resolved absolute paths for later use
        for img_path_str in top_level_images:
            if not img_path_str:
                continue
            
            # Resolve image path
            if os.path.isabs(img_path_str):
                image_path = Path(img_path_str)
            else:
                image_path = base_dir / img_path_str
            
            if image_path.exists() and image_path.is_file():
                try:
                    # Load PIL Image to verify it exists
                    pil_image = PILImage.open(image_path)
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert("RGB")
                    pil_image.load()  # Load immediately to avoid lazy loading issues
                    processed_top_images.append(pil_image)
                    # Store the resolved absolute path for later use in dataset creation
                    processed_top_image_paths.append(str(image_path.resolve()))
                except Exception as e:
                    print(f"Warning: Failed to load top-level image {image_path}: {type(e).__name__}: {e}")
                    missing_images += 1
            else:
                if not image_path.exists():
                    print(f"Warning: Top-level image not found: {image_path}")
                missing_images += 1
        
        # Process messages and their images - store both PIL Image and original path
        processed_messages = []
        processed_msg_image_paths = []  # Store resolved absolute paths from messages
        for msg_idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                print(f"Warning: Skipping message {msg_idx} in item {idx} - not a dictionary")
                continue
            
            processed_msg = {
                "role": msg.get("role", ""),
                "content": msg.get("content", ""),
                "images": []
            }
            
            # Process images in message
            msg_images = msg.get("images", [])
            for img_path_str in msg_images:
                if not img_path_str:
                    continue
                
                # Resolve image path
                if os.path.isabs(img_path_str):
                    image_path = Path(img_path_str)
                else:
                    image_path = base_dir / img_path_str
                
                if image_path.exists() and image_path.is_file():
                    try:
                        # Load PIL Image to verify it exists
                        pil_image = PILImage.open(image_path)
                        if pil_image.mode != 'RGB':
                            pil_image = pil_image.convert("RGB")
                        pil_image.load()  # Load immediately to avoid lazy loading issues
                        processed_msg["images"].append(pil_image)
                        # Store the resolved absolute path for later use in dataset creation
                        processed_msg_image_paths.append(str(image_path.resolve()))
                    except Exception as e:
                        print(f"Warning: Failed to load message image {image_path}: {type(e).__name__}: {e}")
                        missing_images += 1
                else:
                    if not image_path.exists():
                        print(f"Warning: Message image not found: {image_path}")
                    missing_images += 1
            
            processed_messages.append(processed_msg)
        
        # Skip items with no images (both top-level and in messages)
        # Hugging Face requires at least one image for multimodal datasets
        has_images = len(processed_top_images) > 0
        if not has_images:
            # Check if any message has images
            for msg in processed_messages:
                if msg.get("images") and len(msg["images"]) > 0:
                    has_images = True
                    break
        
        if not has_images:
            # Don't print warning for every missing image to avoid spam
            # tqdm will show progress anyway
            missing_images += 1
            continue
        
        # Create processed item - store both PIL Images and paths
        # We'll use paths when creating the dataset, but PIL Images are loaded for validation
        processed_item = {
            "messages": processed_messages,
            "images": processed_top_images,
            "_image_paths": processed_top_image_paths,  # Store paths for later use
            "_msg_image_paths": processed_msg_image_paths  # Store message image paths
        }
        
        processed_data.append(processed_item)
    
    print(f"  Processed {len(processed_data)} valid items")
    if missing_images > 0:
        print(f"  Warning: {missing_images} items skipped due to missing images")
    
    return processed_data


def create_custom_dataset(csv_path, base_dir=None, schema="v3"):
    """
    Create dataset from a single CSV file.
    
    Args:
        csv_path: Path to CSV file
        base_dir: Base directory for resolving image paths (default: CSV file's directory)
        
    Returns:
        DatasetDict with a single "train" split
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    if base_dir is None:
        base_dir = csv_path.parent
    else:
        base_dir = Path(base_dir)
    
    # Define consistent features schema
    # Check if CSV has Subtype and MultiScenario columns
    df_sample = pd.read_csv(csv_path, nrows=1)
    features_dict = {
        "Category": Value("string"),
        "Subcategory": Value("string"),
        "Type": Value("string"),
        "ID": Value("string"),
        "Situation": Value("string"),
        "Action": Value("string"),
        "Risk": Value("string"),
        "Risk Type": Value("string"),
        "Related Hazard": Value("string"),
        "Mitigate Action": Value("string"),
        "URL": Value("string"),
        "image": Image(decode=True),
    }
    
    # Add optional columns if they exist
    if "Subtype" in df_sample.columns:
        features_dict["Subtype"] = Value("string")
    if "MultiScenario" in df_sample.columns:
        features_dict["MultiScenario"] = Value("string")
    
    features = Features(features_dict)
    
    print(f"Loading dataset from {csv_path}...")
    data = load_csv_with_images(csv_path, base_dir)
    dataset = Dataset.from_list(data, features=features)
    if schema == "v3":
        dataset = _normalize_public_schema(dataset, split_name="train")
    print(f"  Loaded {len(data)} examples")
    
    return DatasetDict({"train": dataset})


def create_json_dataset(json_path, base_dir=None):
    """
    Create dataset from a JSON file (OpenAI format training data).
    
    Args:
        json_path: Path to JSON file
        base_dir: Base directory for resolving image paths (default: project root)
        
    Returns:
        DatasetDict with a single "train" split
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    if base_dir is None:
        base_dir = get_project_root()
    else:
        base_dir = Path(base_dir)
    
    # Define features for OpenAI format training data
    # messages is a list of dicts with role and content
    # images is a list of PIL Images
    # Note: load_json_with_images already filters out items with no images
    from datasets import Sequence
    
    print(f"Loading dataset from {json_path}...")
    data = load_json_with_images(json_path, base_dir)
    
    if not data:
        raise ValueError("No valid data found after filtering. All items may have been missing images.")
    
    # Transform data to match Hugging Face's expected format (like the working example)
    # Convert images to single path string (not list) and remove images from messages
    transformed_data = []
    import tempfile
    import json
    
    # Create temp directory for images if we need to save PIL Images
    temp_image_dir = None
    
    for i, item in enumerate(data):
        # Get the first image path - prefer stored paths (original image paths)
        # First try to get from stored paths (these are the original absolute paths)
        image_paths = item.get("_image_paths", [])
        if not image_paths:
            # Try to get from message image paths
            image_paths = item.get("_msg_image_paths", [])
        
        if image_paths:
            # Use the stored original path (absolute path to original image)
            # This ensures Hugging Face can find and upload the image
            image_path = image_paths[0]
            # Verify the image file exists
            if not Path(image_path).exists():
                print(f"Warning: Image path does not exist: {image_path}, skipping item {i}")
                continue
        else:
            # Fallback: get from images list (PIL Images) - shouldn't happen if code above worked
            images_list = item.get("images", [])
            if not images_list:
                # Try to get from messages
                for msg in item.get("messages", []):
                    msg_images = msg.get("images", [])
                    if msg_images:
                        images_list = msg_images
                        break
            
            if not images_list:
                continue  # Skip items with no images
            
            # Use first image - if it's a PIL Image, we don't have the original path
            # This shouldn't happen, but if it does, we'll skip it
            first_image = images_list[0]
            if isinstance(first_image, PILImage.Image):
                print(f"Warning: Item {i} has PIL Image but no stored path, skipping")
                continue
            else:
                # Already a path string
                image_path = str(first_image)
        
        # Create transformed item: remove images from messages, use single image path
        transformed_item = {
            "messages": [
                {
                    "role": msg.get("role", ""),
                    "content": msg.get("content", "")
                    # Remove images from messages to match working example format
                }
                for msg in item.get("messages", [])
            ],
            "images": image_path  # Single path string, not list (like working example)
        }
        transformed_data.append(transformed_item)
    
    # Use Dataset.from_json approach (like the working example)
    # Save to temp JSON file first
    temp_json = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
    json.dump(transformed_data, temp_json, indent=2, ensure_ascii=False)
    temp_json.close()
    
    try:
        # Use from_json which auto-infers features (like the working example)
        # This will automatically process image paths and upload images to Hugging Face
        dataset = Dataset.from_json(temp_json.name)
        print(f"  Loaded {len(transformed_data)} examples")
        
        # Verify images are included
        sample = dataset[0]
        if "images" in sample:
            img_val = sample["images"]
            print(f"  Sample image type: {type(img_val)}")
            if isinstance(img_val, str):
                print(f"  Sample image path: {img_val}")
                if Path(img_val).exists():
                    print(f"  ✓ Image file exists and will be uploaded")
                else:
                    print(f"  ⚠ Warning: Image file not found at path: {img_val}")
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_json.name)
        except:
            pass
    
    return DatasetDict({"train": dataset})


def upload_dataset(dataset_dict, org_name, dataset_name, private=False, token=None, dry_run=False):
    """
    Upload dataset to Hugging Face.
    
    Args:
        dataset_dict: DatasetDict to upload
        org_name: Hugging Face organization name
        dataset_name: Name of the dataset
        private: Whether the dataset should be private
        token: Hugging Face token
    """
    repo_id = f"{org_name}/{dataset_name}"
    
    print(f"\nUploading dataset to {repo_id}...")
    print(f"  Private: {private}")
    print(f"  Splits: {list(dataset_dict.keys())}")
    if dry_run:
        print("  Dry run: skipping push_to_hub")
        for split_name, dataset in dataset_dict.items():
            print(f"  - {split_name}: {len(dataset)} rows, columns={dataset.column_names}")
        return
    
    try:
        dataset_dict.push_to_hub(
            repo_id=repo_id,
            private=private,
            token=token
        )
        print(f"\n✓ Successfully uploaded {dataset_name} to {repo_id}")
        print(f"  View at: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"\n✗ Error uploading {dataset_name}: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Upload datasets to Hugging Face"
    )
    parser.add_argument(
        "--org",
        type=str,
        required=True,
        help="Hugging Face organization name"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["EMBGuardTest", "heldout_set", "both"],
        default=None,
        help="Which predefined dataset(s) to upload (EMBGuardTest, heldout_set, or both)"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Path to custom CSV file to upload (alternative to --dataset)"
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default=None,
        help="Path to JSON file (OpenAI format training data) to upload (alternative to --dataset or --csv-path)"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Name for the dataset when using --csv-path or --json-path (required if --csv-path or --json-path is used)"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base directory for resolving image paths when using --csv-path or --json-path (default: project root for JSON, CSV file's directory for CSV)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private"
    )
    parser.add_argument(
        "--test-split-names",
        choices=["full", "legacy"],
        default="full",
        help="Split naming for EMBGuardTest uploads: full uses causal_risky/etc.; legacy uses HR/HNR/MHR/NHR (default: full)"
    )
    parser.add_argument(
        "--schema",
        choices=["v3", "legacy"],
        default="v3",
        help="Dataset schema for CSV uploads. v3 writes normalized public columns; legacy keeps original CSV columns (default: v3)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build datasets and print split/schema information without uploading"
    )
    parser.add_argument(
        "--token",
        type=str,
        help="Hugging Face token (or set HF_TOKEN environment variable)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if (args.csv_path or args.json_path) and not args.dataset_name:
        print("Error: --dataset-name is required when using --csv-path or --json-path")
        sys.exit(1)
    
    if args.csv_path and args.json_path:
        print("Error: Cannot use both --csv-path and --json-path. Use one or the other.")
        sys.exit(1)
    
    if args.csv_path and args.dataset:
        print("Error: Cannot use both --csv-path and --dataset. Use one or the other.")
        sys.exit(1)
    
    if args.json_path and args.dataset:
        print("Error: Cannot use both --json-path and --dataset. Use one or the other.")
        sys.exit(1)
    
    if not args.csv_path and not args.json_path and not args.dataset:
        print("Error: Must specify either --dataset, --csv-path, or --json-path")
        sys.exit(1)
    
    # Login to Hugging Face
    token = args.token or os.getenv("HF_TOKEN")
    if args.dry_run:
        token = token or None
        print("Dry run enabled: Hugging Face login is not required.")
    elif not token:
        # Try to use huggingface-cli login if available
        try:
            from huggingface_hub import whoami
            try:
                whoami()
                print("Using existing huggingface-cli login")
                token = None  # Will use cached token
            except Exception:
                print("Error: Hugging Face token required.")
                print("Please:")
                print("  1. Set HF_TOKEN environment variable: export HF_TOKEN='your_token'")
                print("  2. Use --token argument: --token your_token")
                print("  3. Or login with: huggingface-cli login")
                sys.exit(1)
        except ImportError:
            print("Error: Hugging Face token required. Set HF_TOKEN environment variable or use --token")
            sys.exit(1)
    
    if token and not args.dry_run:
        login(token=token)
    # If token is None, we're using cached credentials from huggingface-cli login
    
    project_root = get_project_root()
    
    # Handle custom JSON upload (OpenAI format training data)
    if args.json_path:
        print(f"\n{'='*60}")
        print(f"Processing JSON training dataset: {args.dataset_name}")
        print(f"{'='*60}")
        
        json_path = Path(args.json_path)
        if not json_path.is_absolute():
            json_path = project_root / json_path
        
        if not json_path.exists():
            print(f"Error: {json_path} does not exist")
            sys.exit(1)
        
        try:
            base_dir = args.base_dir
            if base_dir:
                if not Path(base_dir).is_absolute():
                    base_dir = project_root / base_dir
            else:
                # Default to project root since JSON image paths are relative to project root
                base_dir = project_root
            
            dataset_dict = create_json_dataset(json_path, base_dir=base_dir)
            upload_dataset(dataset_dict, args.org, args.dataset_name, private=args.private, token=token, dry_run=args.dry_run)
            
        except Exception as e:
            print(f"Error processing JSON dataset: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        # JSON upload completed
        print(f"\n{'='*60}")
        print("Upload process completed!")
        print(f"{'='*60}")
        return
    
    # Handle custom CSV upload
    elif args.csv_path:
        print(f"\n{'='*60}")
        print(f"Processing custom CSV dataset: {args.dataset_name}")
        print(f"{'='*60}")
        
        csv_path = Path(args.csv_path)
        if not csv_path.is_absolute():
            csv_path = project_root / csv_path
        
        if not csv_path.exists():
            print(f"Error: {csv_path} does not exist")
            sys.exit(1)
        
        try:
            base_dir = args.base_dir
            if base_dir:
                if not Path(base_dir).is_absolute():
                    base_dir = project_root / base_dir
            else:
                # Default to project root since CSV image paths are relative to project root
                base_dir = project_root
            
            dataset_dict = create_custom_dataset(csv_path, base_dir=base_dir, schema=args.schema)
            upload_dataset(dataset_dict, args.org, args.dataset_name, private=args.private, token=token, dry_run=args.dry_run)
            
        except Exception as e:
            print(f"Error processing custom dataset: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        # CSV upload completed
        print(f"\n{'='*60}")
        print("Upload process completed!")
        print(f"{'='*60}")
        return
    
    # Handle predefined datasets (only if not using JSON or CSV)
    if args.dataset:
        datasets_to_upload = []
        if args.dataset in ["EMBGuardTest", "both"]:
            datasets_to_upload.append(("EMBGuardTest", "data/test_set"))
        if args.dataset in ["heldout_set", "both"]:
            datasets_to_upload.append(("heldout_set", "data/heldout_set"))
        
        for dataset_name, data_path in datasets_to_upload:
            print(f"\n{'='*60}")
            print(f"Processing {dataset_name}")
            print(f"{'='*60}")
            
            full_data_path = project_root / data_path
            
            if not full_data_path.exists():
                print(f"Error: {full_data_path} does not exist")
                continue
            
            try:
                if dataset_name == "EMBGuardTest":
                    dataset_dict = create_embguardtest_dataset(
                        full_data_path,
                        test_split_names=args.test_split_names,
                        schema=args.schema,
                    )
                elif dataset_name == "heldout_set":
                    dataset_dict = create_heldout_dataset(full_data_path, schema=args.schema)
                else:
                    continue
                
                upload_dataset(dataset_dict, args.org, dataset_name, private=args.private, token=token, dry_run=args.dry_run)
                
            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n{'='*60}")
        print("Upload process completed!")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
