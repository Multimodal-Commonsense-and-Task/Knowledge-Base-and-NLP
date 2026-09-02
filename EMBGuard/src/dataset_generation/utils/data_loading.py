"""
Data loading utilities for dataset generation tasks.
Extracted from inference.py for better modularity.
"""
import json
import os
import sys
from typing import Optional, Dict, Any

# Add project root to Python path (needed before importing utils)
current_dir = os.path.dirname(os.path.abspath(__file__))
# current_dir is src/dataset_generation, so we need to go up 2 levels to get project root
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now we can import utils - use try/except for safety
try:
    from utils.path import get_project_path
    from utils.config import get_config
except ImportError:
    # Fallback: define get_project_path locally if utils.path is not available
    from pathlib import Path
    def get_project_path():
        return Path(__file__).parents[2]  # Go up from src/dataset_generation/data_loading.py to project root
    def get_config():
        return {}


def _resolve_resource_path(data_path: str) -> str:
    """Resolve resource file path using config.yaml or fallback to default"""
    if os.path.isabs(data_path):
        return data_path
    
    # Try to get resources_dir from config.yaml
    config = get_config()
    dataset_gen_config = config.get("dataset_generation", {})
    resources_dir = dataset_gen_config.get("resources_dir", "src/dataset_generation/resources")
    
    # Use get_project_path() from utils.path
    project_root = get_project_path()
    
    # If resources_dir is relative, join with project root
    from pathlib import Path
    if not os.path.isabs(resources_dir):
        resources_path = Path(project_root) / resources_dir
    else:
        resources_path = Path(resources_dir)
    
    return str(resources_path / f'{data_path}.json')


def load_data(data_path):
    file_path = _resolve_resource_path(data_path)
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    print("=========== dataset statistics ===========")
    print(len(data))
    print("==========================================")
    return data


def load_taxonomy_data(data_path, risk_type=None, mechanism=None):
    file_path = _resolve_resource_path(data_path)
    
    with open(file_path, 'r') as f:
        datasets = json.load(f)
    
    # Normalize risk_type: convert "fire risk" to "fire_risk" for matching
    def _normalize(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        return value.replace(" ", "_")

    target_key = _normalize(risk_type)
    
    # Handle both dict and list formats
    if isinstance(datasets, dict):
        items = datasets.items()
        # Filter by risk_type if specified
        if target_key:
            items = [(key, value) for key, value in items if key == target_key]
        
        processed_data = []
        for key, value in items:
            # Get all mechanisms from taxonomy.json
            mechanisms = value.get("mechanism", [])
            
            # Filter by mechanism if specified
            if mechanism:
                filtered_mechanisms = [m for m in mechanisms if m.get("name") == mechanism]
                if not filtered_mechanisms:
                    continue  # Skip this risk type if mechanism not found
                mechanisms = filtered_mechanisms
            
            # Create a separate entry for each mechanism
            for mech in mechanisms:
                processed_data.append({
                    "type": key,
                    "description": value.get("description", "unknown_description"),
                    "mechanism": [mech],  # Each entry contains only one mechanism
                })
    else:
        # List format
        processed_list = datasets
        # Filter by risk_type if specified
        if risk_type:
            normalized_type = _normalize(risk_type)
            processed_list = [t for t in processed_list if _normalize(t.get("type", "")) == normalized_type]
        
        processed_data = []
        for taxonomy in processed_list:
            mechanisms = taxonomy.get("mechanism", [])
            # Create a separate entry for each mechanism
            for mech in mechanisms:
                processed_data.append({
                    "type": taxonomy.get("type", "unknown_type"),
                    "description": taxonomy.get("description", "unknown_description"),
                    "mechanism": [mech],  # Each entry contains only one mechanism
                })
    
    print("=========== dataset statistics ===========")
    print(len(processed_data))
    print("==========================================")
    return processed_data


def load_scenario_data(data_path):
    file_path = _resolve_resource_path(data_path)
    
    with open(file_path, 'r') as f:
        datasets = json.load(f)['scenarios']
        
    processed_data = [
        {
            "id": scenario.get("id", "unknown_id"),
            "risk_type": scenario.get("risk_type", "unknown_risk_type"),
            "mechanism": scenario.get("mechanism", "unknown_mechanism"),
            "hazard": scenario.get("hazard", "unknown_hazard"),
            "action": scenario.get("action", "unknown_action"),
            "source": scenario.get("source", "unknown_source"),
        }
        for scenario in datasets
    ]
    
    print("=========== dataset statistics ===========")
    print(len(processed_data))
    print("==========================================")
    return processed_data


def load_graph_data(data_path):
    # If absolute path or file exists, use it directly
    # Otherwise, resolve as resource path
    if os.path.isabs(data_path) or os.path.exists(data_path):
        file_path = data_path
    else:
    file_path = _resolve_resource_path(data_path)
    
    with open(file_path, 'r') as f:
        datasets = json.load(f)['scenarios']
        
    processed_data = [
        {
            "id": scenario.get("id", ""),
            "risk_type": scenario.get("risk_type", ""),
            "mechanism": scenario.get("mechanism", ""),
            "hazard": scenario.get("hazard", ""),
            "action": scenario.get("action", ""),
            "source": scenario.get("source", ""),
            "merge_source": scenario.get("merge_source", ""),
            "graph": scenario.get("graph", [])
        }
        for scenario in datasets
    ]
    
    print("=========== dataset statistics ===========")
    print(len(processed_data))
    print("==========================================")
    return processed_data


def load_graph_pairs_data(data_path):
    file_path = _resolve_resource_path(data_path)
    
    with open(file_path, 'r') as f:
        datasets = json.load(f)['pairs']
        
    processed_data = [
        {
            "id": pair.get("scenario1_id", "") + "-" + pair.get("scenario2_id", ""),
            "same_risk_type": pair.get("same_risk_type", False),
            "risk_combination": pair.get("risk_combination", ""),
            "graph1": pair.get("scenario1_data", {}).get("graph", []),
            "hazard1": pair.get("scenario1_data", {}).get("hazard", ""),
            "action1": pair.get("scenario1_data", {}).get("action", ""),
            "graph2": pair.get("scenario2_data", {}).get("graph", []),
            "hazard2": pair.get("scenario2_data", {}).get("hazard", ""),
            "action2": pair.get("scenario2_data", {}).get("action", ""),
        }
        for pair in datasets
    ]
    
    print("=========== dataset statistics ===========")
    print(len(processed_data))
    print("==========================================")
    return processed_data


def load_text_data(data_path):
    """Load text data (graph_to_text results with situation field)"""
    file_path = _resolve_resource_path(data_path)
    
    with open(file_path, 'r') as f:
        datasets = json.load(f)['scenarios']

    # Keep full scenario payload so batch jobs can retain rich metadata
    # (e.g., hazard_augmented dual fields like hazard1/2, pair_info, inference_info).
    processed_data = []
    for scenario in datasets:
        if not isinstance(scenario, dict):
            continue
        scenario_copy = scenario.copy()
        scenario_copy.setdefault("id", "")
        scenario_copy.setdefault("merge_source", "")
        scenario_copy.setdefault("situation", "")
        processed_data.append(scenario_copy)
    
    print("=========== dataset statistics ===========")
    print(len(processed_data))
    print("==========================================")
    return processed_data


def load_vqa_data(data_path):
    """Load image QA data (question + image pairs)"""
    file_path = _resolve_resource_path(data_path)
    
    with open(file_path, 'r') as f:
        datasets = json.load(f)['scenarios']
        
    processed_data = []

    def add_record(question: str, image_path: str, extra: Dict[str, Any]) -> None:
        if not question:
            return
        record = {
            "question": question,
            "image_path": image_path or extra.get("image_path", "") or extra.get("image", ""),
        }
        for key in ["scenario_id", "risk_type", "mechanism", "expected_answer"]:
            if extra.get(key):
                record[key] = extra[key]
        processed_data.append(record)
    
    for scenario in datasets:
        if not isinstance(scenario, dict):
            continue
        image_path = scenario.get("image_path")
        
        base_extra = {
            "scenario_id": scenario.get("id"),
            "risk_type": scenario.get("risk_type"),
            "mechanism": scenario.get("mechanism"),
        }
        
        qa_pairs = scenario.get("qa_pairs")
        
        for pair in qa_pairs:
            if not isinstance(pair, dict):
                continue
            question = pair.get("question", "")
            extra = dict(base_extra)
            extra["expected_answer"] = pair.get("answer")
            add_record(question, image_path, extra)
    
    print("=========== dataset statistics ===========")
    print(len(processed_data))
    print("==========================================")
    
    return processed_data


def load_guardrail_data(data_path):
    """Load guardrail data (action + image pairs)"""
    file_path = _resolve_resource_path(data_path)
    
    with open(file_path, 'r') as f:
        datasets = json.load(f)['scenarios']
        
    processed_data = []
    
    for scenario in datasets:
        if not isinstance(scenario, dict):
            continue
        
        # Get image_path and action from scenario
        image_path = scenario.get("image_path")
        action = scenario.get("action")
        
        # Skip scenarios without image_path (only process those with images)
        if not image_path or not action:
            continue
        
        processed_data.append({
            "id": scenario.get("id", ""),
            "risk_type": scenario.get("risk_type", ""),
            "mechanism": scenario.get("mechanism", ""),
            "hazard": scenario.get("hazard", ""),
            "action": action,
            "image_path": image_path,
        })
    
    print("=========== dataset statistics ===========")
    print(len(processed_data))
    print("==========================================")
    
    return processed_data

