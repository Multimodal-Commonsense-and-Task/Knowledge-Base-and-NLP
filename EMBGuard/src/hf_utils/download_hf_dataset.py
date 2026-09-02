#!/usr/bin/env python3
"""
Download Hugging Face dataset with images to local directory.

This script downloads a Hugging Face dataset and saves it as original files
(CSV with image files) rather than in Hugging Face's internal format.

Usage:
    python -m src.hf_utils.download_hf_dataset \
        --dataset-id EMBGuard/EMBHazard_original_wo_filter_v1.0 \
        --output-dir /path/to/output \
        [--split train] \
        [--token YOUR_TOKEN]
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from utils.test_types import hf_split_candidates, normalize_test_type_code

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError(
        "datasets library is required. Install it via: pip install datasets"
    )


def download_dataset_with_images(
    dataset_id: str,
    output_dir: Path,
    split: Optional[str] = None,
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> None:
    """
    Download Hugging Face dataset with images and save as original files.
    
    Args:
        dataset_id: Hugging Face dataset ID (e.g., "EMBGuard/EMBHazard_original_wo_filter_v1.0")
        output_dir: Directory to save the dataset
        split: Dataset split to download (e.g., "train"). If None, downloads all splits.
        token: Hugging Face token for authentication (optional)
        cache_dir: Cache directory for datasets library (optional)
    """
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split = normalize_test_type_code(split, default=split)
    
    print(f"Downloading dataset: {dataset_id}")
    print(f"Output directory: {output_dir}")
    
    # Load dataset
    if split:
        print(f"Loading split: {split}")
        last_error = None
        for split_candidate in hf_split_candidates(split):
            try:
                dataset = load_dataset(
                    dataset_id,
                    split=split_candidate,
                    token=token,
                    cache_dir=cache_dir,
                )
                split = split_candidate
                break
            except Exception as e:
                last_error = e
        else:
            raise last_error
        splits = {split: dataset}
    else:
        print("Loading all splits...")
        dataset_dict = load_dataset(
            dataset_id,
            token=token,
            cache_dir=cache_dir,
        )
        splits = dataset_dict
    
    # Process each split
    for split_name, split_dataset in splits.items():
        print(f"\nProcessing split: {split_name}")
        print(f"  Total examples: {len(split_dataset)}")
        
        # Create output directory for this split
        split_output_dir = output_dir / split_name
        split_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create images directory
        images_dir = split_output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect data for CSV
        csv_data = []
        
        # Process each example
        for idx, example in tqdm(
            enumerate(split_dataset),
            total=len(split_dataset),
            desc=f"Processing {split_name}",
            unit="examples"
        ):
            row = {}
            
            # Extract all fields except image
            for key, value in example.items():
                if key == 'image':
                    # Handle image separately
                    continue
                # Convert value to string, handle None
                if value is None:
                    row[key] = ''
                else:
                    row[key] = str(value)
            
            # Handle image
            if 'image' in example and example['image'] is not None:
                try:
                    image = example['image']
                    
                    # Get image filename from a path-like column if available.
                    image_filename = None
                    image_path_value = (
                        row.get('image_path', '')
                        or row.get('source_path', '')
                        or row.get('URL', '')
                        or row.get('url', '')
                    )
                    if image_path_value:
                        image_filename = Path(image_path_value).name
                    
                    # If no filename is available from image_path/source_path, generate one.
                    if not image_filename:
                        # Try to infer from image format
                        if hasattr(image, 'format') and image.format:
                            ext = image.format.lower()
                            if ext == 'jpeg':
                                ext = 'jpg'
                            image_filename = f"image_{idx:06d}.{ext}"
                        else:
                            image_filename = f"image_{idx:06d}.jpg"
                    
                    # Ensure unique filename
                    image_path = images_dir / image_filename
                    counter = 1
                    while image_path.exists():
                        stem = Path(image_filename).stem
                        suffix = Path(image_filename).suffix
                        image_filename = f"{stem}_{counter}{suffix}"
                        image_path = images_dir / image_filename
                        counter += 1
                    
                    # Save image
                    from PIL import Image as PILImage
                    
                    if isinstance(image, PILImage.Image):
                        # Already a PIL Image
                        image.save(image_path)
                    elif hasattr(image, 'save'):
                        # Has save method (might be PIL Image wrapper)
                        image.save(image_path)
                    else:
                        # Try to convert to PIL Image
                        try:
                            pil_image = PILImage.fromarray(image) if hasattr(image, '__array__') else PILImage.open(image)
                            pil_image.save(image_path)
                        except Exception as e:
                            print(f"Warning: Could not save image for example {idx}: {e}")
                            image_filename = None
                    
                    # Update image_path to the downloaded relative path.
                    if image_filename:
                        relative_image_path = f"images/{image_filename}"
                        row['image_path'] = relative_image_path
                
                except Exception as e:
                    print(f"Warning: Failed to process image for example {idx}: {e}")
                    # Continue without image
                    if 'image_path' not in row:
                        row['image_path'] = ''
            
            csv_data.append(row)
        
        # Save CSV
        import pandas as pd
        csv_path = split_output_dir / "dataset.csv"
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"  Saved CSV: {csv_path}")
        print(f"  Saved {len([r for r in csv_data if r.get('image_path')])} images")
    
    print(f"\n✓ Dataset downloaded successfully to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Hugging Face dataset with images to local directory"
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        required=True,
        help="Hugging Face dataset ID (e.g., EMBGuard/EMBHazard_original_wo_filter_v1.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory to save the dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split to download (e.g., 'train'). If not specified, downloads all splits.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token for authentication. Can also use HF_TOKEN or HUGGINGFACE_TOKEN env vars.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for datasets library. If not specified, uses default cache.",
    )
    
    args = parser.parse_args()
    
    # Get token from argument or environment
    token = args.token
    if token is None:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    
    # Get cache directory from environment if not specified
    cache_dir = args.cache_dir
    if cache_dir is None:
        cache_dir = os.getenv("HF_DATASETS_CACHE") or os.getenv("HF_HOME")
        if cache_dir and not cache_dir.endswith("/datasets"):
            cache_dir = os.path.join(cache_dir, "datasets")
    
    download_dataset_with_images(
        dataset_id=args.dataset_id,
        output_dir=Path(args.output_dir),
        split=args.split,
        token=token,
        cache_dir=cache_dir,
    )


if __name__ == "__main__":
    main()
