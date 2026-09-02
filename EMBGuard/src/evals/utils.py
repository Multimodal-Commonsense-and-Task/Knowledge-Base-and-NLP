"""
Common utilities for evaluation modules
"""
import os
import warnings
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any, Union
from datasets import load_dataset

from utils.config import get_config
from utils.path import get_project_path
from utils.test_types import hf_split_candidates, normalize_test_type_code


def load_data(data_source: str, split: Optional[str] = None) -> Tuple[pd.DataFrame, bool, Optional[Path]]:
    """
    Load data from CSV file or Hugging Face dataset
    
    Args:
        data_source: Path to CSV file or Hugging Face dataset name (e.g., "org/dataset_name")
        split: Split name for Hugging Face dataset. EMBGuardTest accepts aliases
            such as "HR", "causal_risky", or "Causal Risky"; heldout sets use
            "safe" or "unsafe".
        
    Returns:
        Tuple of (DataFrame, is_hf_dataset, csv_dir)
        - DataFrame: DataFrame with test data
        - is_hf_dataset: True if loaded from Hugging Face, False if from CSV
        - csv_dir: Directory containing CSV (None for HF datasets)
    """
    split = normalize_test_type_code(split, default=split)

    # Check if it's a Hugging Face dataset (contains "/" and doesn't exist as file)
    if "/" in data_source and not Path(data_source).exists():
        # Try to load from Hugging Face Hub
        try:
            # Suppress verbose output from datasets library
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
                    # Try to get the split from the dataset
                    if isinstance(dataset_dict, dict):
                        # If multiple splits, use the first one or specified split
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


def resolve_image(item: Dict[str, Any], csv_dir: Optional[Path], is_hf_dataset: bool) -> Union[str, Path]:
    """
    Resolve image from image_path or Hugging Face dataset
    
    Args:
        item: Row/item dictionary containing image information
        csv_dir: Directory containing the CSV file (None for HF datasets)
        is_hf_dataset: Whether this is from Hugging Face dataset
        
    Returns:
        Image path (str or Path)
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
        # For CSV files, resolve image path from normalized image_path or legacy aliases.
        image_path_value = (
            item.get("image_path", "")
            or item.get("source_path", "")
            or item.get("URL", "")
            or item.get("url", "")
            or item.get("path", "")
        )
        if not image_path_value:
            raise ValueError("No image path found in CSV row")
        
        # Get data_dir from config
        config = get_config()
        common_config = config.get("common", {})
        data_dir = common_config.get("data_dir", "data/test_set")
        
        # Resolve data_dir to absolute path
        project_path = get_project_path()
        if not Path(data_dir).is_absolute():
            data_dir_path = project_path / data_dir
        else:
            data_dir_path = Path(data_dir)
        
        # Resolve image path relative to data_dir
        image_path = data_dir_path / image_path_value
        if not image_path.exists():
            # Fallback: try resolving from CSV directory
            if csv_dir:
                image_path = csv_dir / image_path_value
                if not image_path.exists():
                    raise FileNotFoundError(f"Image not found: {image_path_value} (tried: {data_dir_path / image_path_value}, {csv_dir / image_path_value})")
            else:
                raise FileNotFoundError(f"Image not found: {image_path_value} (tried: {data_dir_path / image_path_value})")
        return image_path


def convert_messages_for_storage(messages: List[Dict[str, Any]], image_path: str) -> List[Dict[str, Any]]:
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
