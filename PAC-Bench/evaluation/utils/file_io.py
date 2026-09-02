"""
File I/O Utilities

File load/save utility (commonly used in all stages)

Output directory structure: (time)/(domain)/(stage_name)_#.json
"""

import json
from pathlib import Path
from typing import Any, Callable, Optional


def sanitize_domain_name(domain_name: str) -> str:
    """Convert domain name to a form safe for the file system"""
    return domain_name.replace(" ", "_").replace("/", "_").replace(",", "").replace("\\", "_")


def extract_domain_from_path(path: Path) -> str:
    """
    Extract domain name from file path
    
    Folder structure: .../timestamp/domain_name/file.json
    -> Extract from parent folder name (domain_name)
    """
    return path.parent.name.replace("_", " ")


def extract_index_from_filename(path: Path, prefix: str) -> int:
    """
    Extract index from filename
    
    Example: scenario_0.json -> 0, requirements_3.json -> 3
    """
    stem = path.stem  # scenario_0, criteria_3, etc.
    if stem.startswith(f"{prefix}_"):
        try:
            return int(stem.split("_", 1)[1])
        except (ValueError, IndexError):
            pass
    return 0


def save_stage_result(
    output_dir: Path,
    domain_name: str,
    stage_prefix: str,
    index: int,
    data: dict,
) -> Path:
    """
    Save stage results in standard directory structure
    
    Structure: output_dir/domain_name/{stage_prefix}_{index}.json
    
    Args:
        output_dir: Output base directory (includes timestamp)
        domain_name: Domain name (used as folder name)
    stage_prefix: Stage prefix (scenario, requirements, context, policy)
        index: File index
        data: JSON data to save
        
    Returns:
        Path: Saved file path
    """
    safe_domain = sanitize_domain_name(domain_name)
    domain_dir = output_dir / safe_domain
    domain_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = domain_dir / f"{stage_prefix}_{index}.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return output_path


def load_stage_files(
    input_path: str,
    file_prefix: str,
) -> list[tuple[str, int, Path, dict]]:
    """
    Load results from previous stages
    
    Folder structure:
        {timestamp}/
            {domain_name}/
                {file_prefix}_0.json
                {file_prefix}_1.json
                ...
    
    Args:
        input_path: Input path (file or directory)
        file_prefix: File prefix (scenario, requirements, context, policy)
        
    Returns:
        list[tuple[str, int, Path, dict]]: List of (domain_name, index, file_path, json_data) tuples
    """
    path = Path(input_path)
    results = []
    
    if path.is_file() and path.suffix == ".json":
        # Single file - extract domain from parent folder name
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        domain_name = extract_domain_from_path(path)
        index = extract_index_from_filename(path, file_prefix)
        results.append((domain_name, index, path, data))
    
    elif path.is_dir():
        # Search for domain folders (subfolders)
        domain_dirs = [d for d in path.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        if domain_dirs:
            # Case of folder structure by domain
            for domain_dir in sorted(domain_dirs):
                files = sorted(domain_dir.glob(f"{file_prefix}_*.json"))
                
                if not files:
                    continue
                
                # Extract domain name from folder name (underscores to spaces)
                domain_name = domain_dir.name.replace("_", " ")
                
                for json_file in files:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    index = extract_index_from_filename(json_file, file_prefix)
                    results.append((domain_name, index, json_file, data))
        else:
            # Legacy: Case where files are directly present without domain folders
            files = sorted(path.glob(f"{file_prefix}_*.json"))
            
            if not files:
                raise ValueError(f"No {file_prefix}_*.json files or domain folders found in {path}")
            
            # Use folder name as domain name
            domain_name = path.name.replace("_", " ")
            
            for json_file in files:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                index = extract_index_from_filename(json_file, file_prefix)
                results.append((domain_name, index, json_file, data))
    
    else:
        raise FileNotFoundError(f"Input not found: {path}")
    
    if not results:
        raise ValueError(f"No {file_prefix} results found in {path}")
    
    return results


def remove_index_from_scenario(data: dict) -> tuple[int, dict]:
    """
    Extract index from Scenario JSON and then remove it
    
    Args:
        data: scenario JSON data
        
    Returns:
        tuple[int, dict]: (extracted index, data with index removed)
    """
    if "scenario" in data and "index" in data["scenario"]:
        index = data["scenario"]["index"]
        del data["scenario"]["index"]
        return index, data
    return 0, data

