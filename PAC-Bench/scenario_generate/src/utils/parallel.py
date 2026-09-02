"""
Parallel Processing Utilities

Parallel processing utilities

Applying SOLID principles:
- SRP: Responsible only for executing parallel processing
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import TypeVar, Callable
from pathlib import Path

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Dummy class to operate even if tqdm is not available
    class tqdm:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, n=1):
            pass
        def write(self, s):
            pass

T = TypeVar('T')


def run_parallel(
    items: list[T],
    process_fn: Callable[[T], tuple[str, bool, str]],
    max_workers: int,
    logger: logging.Logger,
    task_name: str = "task",
) -> tuple[int, int]:
    """
    Parallel processing utility
    
    Args:
        items: List of items to process
        process_fn: Processing function
            - Input: Each item (T)
            - Return: (identifier, success/failure, result path or error message) tuple
        max_workers: Maximum number of workers
        logger: Logger
        task_name: Task name (for logging)
    
    Returns:
        tuple[int, int]: (Success count, Failure count)
    
    Example:
        >>> def process_item(item):
        ...     try:
        ...         result = do_something(item)
        ...         return (item.name, True, result_path)
        ...     except Exception as e:
        ...         return (item.name, False, str(e))
        >>> 
        >>> success, fail = run_parallel(items, process_item, max_workers=4, logger=logger)
    """
    results_lock = Lock()
    success_count = 0
    fail_count = 0
    
    logger.info(f"Starting parallel processing for {len(items)} {task_name}(s) with {max_workers} workers")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(
        total=len(items),
        desc=f"Processing {task_name}s",
        unit=task_name,
        disable=not TQDM_AVAILABLE,
    ) as progress:
        future_to_item = {executor.submit(process_fn, item): item for item in items}
        
        for future in as_completed(future_to_item):
            try:
                name, success, _ = future.result()
                with results_lock:
                    if success:
                        success_count += 1
                    else:
                        fail_count += 1
                progress.update(1)
            except Exception as e:
                logger.error(f"Unexpected error in parallel task: {e}")
                with results_lock:
                    fail_count += 1
                progress.update(1)
    
    return success_count, fail_count


def run_parallel_files(
    files: list[Path],
    process_file_fn: Callable[[Path], tuple[str, bool, str]],
    max_workers: int,
    logger: logging.Logger,
    task_name: str = "file",
) -> tuple[int, int]:
    """
    Helper that processes a list of Path (files) in parallel, one file at a time.
    Internally reuses run_parallel.
    """
    return run_parallel(
        items=files,
        process_fn=process_file_fn,
        max_workers=max_workers,
        logger=logger,
        task_name=task_name,
    )

