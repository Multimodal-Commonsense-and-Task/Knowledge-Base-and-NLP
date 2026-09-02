"""
Multi-process inference utilities for EMBGuardTest
Handles parallel inference execution
"""
import sys
from pathlib import Path
from multiprocessing import Process, Manager
from typing import List, Dict, Any, Callable, Tuple
from tqdm import tqdm

# Add project root to path
# inference_utils.py is in src/utils/, so parent.parent.parent gives project root
# Path structure: project_root/src/utils/inference_utils.py
def _get_project_root():
    """Get project root directory"""
    current_file = Path(__file__)
    if current_file.is_absolute():
        return current_file.resolve().parent.parent.parent
    else:
        # If __file__ is relative, try to resolve it
        try:
            return current_file.resolve().parent.parent.parent
        except:
            # Fallback: assume we're in src/utils/
            return Path.cwd()

project_root = _get_project_root()
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from utils.test_types import normalize_test_type_code, test_type_safety_type, test_type_slug

# Delay imports until they're actually needed
# This avoids import errors when the module is first loaded
# The imports will happen in the functions that use them
EMBGuard = None
get_config = None
get_project_path = None

def _ensure_imports():
    """Ensure required modules are imported"""
    global EMBGuard, get_config, get_project_path
    if EMBGuard is None or get_config is None or get_project_path is None:
        project_root = _get_project_root()
        project_root_str = str(project_root)
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)
        
        try:
            from src.guardrail.guardrail import EMBGuard as _EMBGuard
            from utils.config import get_config as _get_config
            from utils.path import get_project_path as _get_project_path
            EMBGuard = _EMBGuard
            get_config = _get_config
            get_project_path = _get_project_path
        except ImportError as e:
            # Last resort: print error and raise
            print(f"Failed to import modules. Project root: {project_root_str}")
            print(f"sys.path: {sys.path[:5]}")
            print(f"Error: {e}")
            raise


def worker_main(work_queue, result_queue, process_func, config):
    """
    Worker process main function
    
    Args:
        work_queue: Queue containing work items
        result_queue: Queue for results
        process_func: Function to process each item
        config: Configuration dictionary
    """
    while True:
        item = work_queue.get()
        if item is None:
            result_queue.put(None)
            break
        try:
            result, cost, usage = process_func(config, item)
            result_queue.put((result, cost, usage))
        except Exception as e:
            item_info = item.get('idx', item.get('id', 'unknown item'))
            print(f"Error processing item {item_info}: {e}")
            result_queue.put((None, 0.0, {}))
        finally:
            work_queue.task_done()


def process_inference_item(config: Dict[str, Any], item: Dict[str, Any]) -> Tuple[Dict[str, Any], float, Dict[str, int]]:
    """
    Process a single inference item
    
    Args:
        config: Configuration dictionary containing:
            - provider: LLM provider
            - model_config: Model configuration
            - csv_dir: CSV directory for resolving image paths
            - use_few_shot: Whether to use few-shot examples
            - use_thinking: Whether to use thinking mode (step-by-step reasoning)
        item: Dictionary containing:
            - idx: Row index
            - row: DataFrame row as dictionary
            - image_path: Image path from the normalized dataset schema
            - csv_dir: CSV directory path
            
    Returns:
        Tuple of (result_dict, cost, usage_dict)
    """
    # Ensure imports are available (for multiprocessing workers)
    _ensure_imports()
    
    # Initialize EMBGuard in worker process
    provider = config["provider"]
    model_config = config["model_config"]
    guard = EMBGuard(provider, model_config)
    
    # Extract data from item
    idx = item["idx"]
    row = item["row"]
    image_path_value = (
        item.get("image_path", "")
        or item.get("image_url", "")
        or row.get("image_path", "")
        or row.get("source_path", "")
        or row.get("URL", "")
        or row.get("url", "")
        or row.get("path", "")
    )
    csv_dir_str = item.get("csv_dir")
    csv_dir = Path(csv_dir_str) if csv_dir_str else None
    is_hf_dataset = item.get("is_hf_dataset", False)
    use_few_shot = config.get("use_few_shot", True)
    
    # Extract action
    action = row.get("action", "") or row.get("Action", "")
    if not action or action == "":
        raise ValueError("Action field is missing or empty")
    
    # Resolve image (handles both CSV paths and HF dataset images)
    if is_hf_dataset:
        # For Hugging Face datasets, check for "image" column first
        if "image" in row and row["image"] is not None:
            # PIL Image object from Hugging Face (may be dict after to_pandas conversion)
            from PIL import Image as PILImage
            import tempfile
            import io
            
            pil_image = row["image"]
            
            # Handle dict format (from to_pandas conversion)
            if isinstance(pil_image, dict):
                if "bytes" in pil_image and pil_image["bytes"]:
                    pil_image = PILImage.open(io.BytesIO(pil_image["bytes"])).convert("RGB")
                elif "path" in pil_image and pil_image["path"]:
                    pil_image = PILImage.open(pil_image["path"]).convert("RGB")
                else:
                    raise ValueError("Invalid image dict format")
            
            # Create temporary file
            temp_dir = Path(tempfile.gettempdir()) / "embguard_images"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename from ID or index
            image_id = row.get("id") or row.get("ID", f"img_{idx}")
            temp_path = temp_dir / f"{image_id}.jpg"
            
            # Save PIL image to temp file
            if isinstance(pil_image, PILImage.Image):
                pil_image.save(temp_path, "JPEG")
                image_path = temp_path
            else:
                image_path = Path(str(pil_image))
        elif image_path_value:
            # Fallback to a path-like field if the image column is not available.
            image_path = Path(image_path_value)
        else:
            raise ValueError("No image found in Hugging Face dataset item")
    else:
        # For CSV files, resolve image path
        if not image_path_value:
            raise ValueError("No image path found in CSV row")
        image_path = _resolve_image_path(image_path_value, csv_dir)
    
    # Get use_thinking from config
    use_thinking = config.get("use_thinking", False)
    
    # Prepare messages (to save in output)
    # Note: prepare_messages returns messages with image path, not base64
    messages = guard.prepare_messages(
        action=action,
        image=str(image_path),
        use_few_shot=use_few_shot,
        use_thinking=use_thinking
    )
    
    # Convert messages for storage (replace image objects with image_path)
    messages_for_storage = _convert_messages_for_storage(messages, str(image_path))
    
    # Run inference using EMBGuard
    inference_result = guard.evaluate(
        action=action,
        image=str(image_path),
        use_few_shot=use_few_shot,
        use_thinking=use_thinking
    )
    
    # scenario_type is the EMBGuard condition; type is the dataset safety class.
    scenario_code = None
    for scenario_candidate in (
        item.get("scenario_type"),
        config.get("test_set_type"),
        row.get("scenario_type"),
        row.get("Type"),
        row.get("Subtype"),
        item.get("type"),  # legacy result payloads used type for the condition.
    ):
        scenario_code = normalize_test_type_code(scenario_candidate, default=None)
        if scenario_code:
            break
    if not scenario_code:
        scenario_code = "UNKNOWN"

    safety_type = str(
        row.get("type")
        or item.get("type")
        or item.get("dataset_type")
        or config.get("dataset_type")
        or ""
    ).strip().lower()
    if safety_type not in {"safe", "unsafe"}:
        safety_type = test_type_safety_type(scenario_code) or "unknown"
    
    # Build result
    result = {
        "idx": int(idx),
        "type": safety_type,
        "scenario_type": test_type_slug(scenario_code),
        "csv_row": row,
        "action": action,
        "image_path": str(image_path),
        "messages": messages_for_storage,  # Save messages sent to model (with image_path instead of base64)
        "response": inference_result["response"],
        "parsed_response": inference_result["parsed_response"],
        "usage": inference_result["usage"],
        "cost": inference_result["cost"],
    }
    
    # Keep dataset_type only when a legacy caller provided it explicitly.
    dataset_type = item.get("dataset_type") or config.get("dataset_type")
    if dataset_type and dataset_type != safety_type:
        result["dataset_type"] = dataset_type
    
    # Include ID if present
    row_id = row.get("id") or row.get("ID")
    if row_id:
        result["id"] = str(row_id)
    
    return result, inference_result["cost"], inference_result["usage"]


def _convert_messages_for_storage(messages: List[Dict[str, Any]], image_path: str) -> List[Dict[str, Any]]:
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


def _resolve_image_path(image_path_value: str, csv_dir: Path) -> Path:
    """
    Resolve image path using data_dir from config
    
    Args:
        image_path_value: Image path from CSV or dataset metadata
        csv_dir: Directory containing the CSV file
        
    Returns:
        Resolved image path
    """
    # Ensure imports are available
    _ensure_imports()
    
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
        image_path = csv_dir / image_path_value
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path_value} (tried: {data_dir_path / image_path_value}, {csv_dir / image_path_value})")
    return image_path


def run_parallel_inference(
    dataset: List[Dict[str, Any]],
    config: Dict[str, Any],
    num_workers: int = 4,
    description: str = "Running inference",
    process_func: Callable = None
) -> Tuple[List[Dict[str, Any]], float, Dict[str, int]]:
    """
    Run parallel inference on the given dataset
    
    Args:
        dataset: List of items to process. Each item should be a dict with:
            - idx: Row index (or similar identifier)
            - Other fields depend on process_func requirements
        config: Configuration dictionary (contents depend on process_func)
        num_workers: Number of worker processes to use
        description: Description to display on the tqdm progress bar
        process_func: Function to process each item. If None, uses process_inference_item.
            Signature: (config: Dict, item: Dict) -> (result: Dict, cost: float, usage: Dict)
        
    Returns:
        Tuple of (list of results, total_cost, total_usage)
    """
    # Use default process function if not provided
    if process_func is None:
        process_func = process_inference_item
    
    manager = Manager()
    work_queue = manager.Queue()
    result_queue = manager.Queue()
    
    # Add data to the work queue
    for data in dataset:
        work_queue.put(data)
    
    # Add termination signals for workers
    for _ in range(num_workers):
        work_queue.put(None)
    
    # Start parallel processing
    processes = []
    for _ in range(num_workers):
        p = Process(target=worker_main, args=(work_queue, result_queue, process_func, config))
        p.start()
        processes.append(p)
    
    # Show progress bar and collect results
    process_results = []
    process_cost = 0.0
    process_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    completed_workers = 0
    
    with tqdm(total=len(dataset), desc=description) as pbar:
        while completed_workers < num_workers:
            result_item = result_queue.get()
            if result_item is None:
                completed_workers += 1
            else:
                result, cost, usage = result_item
                if result is not None:
                    process_results.append(result)
                    process_cost += cost if cost is not None else 0.0
                    if usage:
                        process_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                        process_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                        process_usage["total_tokens"] += usage.get("total_tokens", 0)
                pbar.update(1)
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    # Collect remaining results
    while not result_queue.empty():
        result_item = result_queue.get_nowait()
        if result_item is not None:
            result, cost, usage = result_item
            if result is not None:
                process_results.append(result)
                process_cost += cost if cost is not None else 0.0
                if usage:
                    process_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                    process_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                    process_usage["total_tokens"] += usage.get("total_tokens", 0)
    
    # Sort results by idx to maintain order
    process_results.sort(key=lambda x: x.get("idx", 0) if isinstance(x, dict) and "idx" in x else 0)
    
    return process_results, process_cost, process_usage
