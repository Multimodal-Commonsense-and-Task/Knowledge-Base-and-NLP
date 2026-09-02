"""
Base task class for dataset generation pipeline.
Each specific task is implemented in tasks/ directory.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import os
import sys

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from utils.path import get_project_path
    from utils.config import get_config
except ImportError:
    from pathlib import Path
    def get_project_path():
        return Path(__file__).parents[2]
    def get_config():
        return {}

from src.dataset_generation.prompters import (
    GPTPrompter, VLMGPTPrompter, 
    RiskPipelinePrompter, GraphPostprocessingPrompter
)


def _resolve_resource_path(data_path: str) -> str:
    """Resolve resource file path using config.yaml or fallback to default"""
    if os.path.isabs(data_path):
        return data_path
    
    # Try to get resources_dir from config.yaml
    config = get_config()
    dataset_gen_config = config.get("dataset_generation", {})
    resources_dir = dataset_gen_config.get("resources_dir", "src/dataset_generation/resources")
    
    project_root = get_project_path()
    
    # If resources_dir is relative, join with project root
    from pathlib import Path
    if not os.path.isabs(resources_dir):
        resources_path = Path(project_root) / resources_dir
    else:
        resources_path = Path(resources_dir)
    
    return str(resources_path / f'{data_path}.json')


class BaseTask(ABC):
    """Base class for all tasks in the dataset generation pipeline."""
    
    def __init__(self, args):
        self.args = args
        self.task_name = self.get_task_name()
    
    @abstractmethod
    def get_task_name(self) -> str:
        """Return the task name (prompt_key)."""
        pass
    
    @abstractmethod
    def load_data(self) -> List[Dict[str, Any]]:
        """Load and prepare input data for this task."""
        pass
    
    @abstractmethod
    def create_prompter(self):
        """Create and return the appropriate prompter for this task."""
        pass
    
    def prepare_model_inputs(self, dataset: List[Dict[str, Any]], prompter) -> List:
        """Prepare model inputs from dataset."""
        all_model_inputs = []
        from tqdm import tqdm
        
        for data in tqdm(dataset, desc="Preparing model inputs"):
            # Add num to input_data for taxonomy_to_scenario
            if self.task_name == 'taxonomy_to_scenario':
                data_with_num = {**data, "num": self.args.num}
            else:
                data_with_num = data
            
            # All providers use GPT-style prompts
            model_input = prompter.build_prompt(data_with_num)
            all_model_inputs.append([model_input, data])
        
        return all_model_inputs
    
    def process_result(self, response: Dict[str, Any], model_input: List) -> Dict[str, Any]:
        """
        Process the LLM response and return structured result.
        
        Args:
            response: Response from models.py BaseLLMModel.generate()
                - content: Generated text content
                - usage: Token usage dict
                - cost: Cost in USD
            model_input: [prompt_string, metadata] tuple
        
        Returns:
            Processed result dictionary
        """
        from src.dataset_generation.utils.json_utils import parse_json_prediction
        
        # Extract content from models.py response format
        # models.py returns {"content": str, "usage": dict, "cost": float}
        content = response.get('content', '')
        if not content:
            # Fallback: try 'prediction' key for backward compatibility
            content = response.get('prediction', '')
        
        json_prompts = [
            "taxonomy_to_scenario", "scenario_to_graph", "scene_normalization",
            "scene_augmentation", "hazard_augmentation", "hazard_removal", 
            "action_augmentation", "qa_generation", "vqa", "guardrail"
        ]
        
        if self.task_name in ["graph_to_image", "text_to_image"]:
            # Image generation tasks - models.py doesn't support this yet
            # Keep original format for now
            return {
                "prediction": content,
                "image_url": response.get('image_url'),
                "all_image_urls": response.get('all_image_urls', []),
                "model_input": model_input[0],
                **model_input[1],
            }
        elif self.task_name == "vqa":
            parsed_json = parse_json_prediction(content)
            if parsed_json is not None:
                result = {
                    "prediction": parsed_json,
                    "raw_prediction": content,
                    "question": model_input[1].get("question", ""),
                    "image_path": model_input[1].get("image_path", ""),
                }
                for key in ["scenario_id", "risk_type", "mechanism", "expected_answer"]:
                    if key in model_input[1]:
                        result[key] = model_input[1][key]
                return result
            else:
                result = {
                    "prediction": content,
                    "parse_error": "Failed to parse JSON",
                    "question": model_input[1].get("question", ""),
                    "image_path": model_input[1].get("image_path", ""),
                }
                for key in ["scenario_id", "risk_type", "mechanism", "expected_answer"]:
                    if key in model_input[1]:
                        result[key] = model_input[1][key]
                return result
        elif self.task_name == "guardrail":
            parsed_json = parse_json_prediction(content)
            if parsed_json is not None:
                result = {
                    "prediction": parsed_json,
                    "raw_prediction": content,
                    "action": model_input[1].get("action", ""),
                    "image_path": model_input[1].get("image_path", ""),
                }
                for key in ["scenario_id", "risk_type", "mechanism", "hazard"]:
                    if key in model_input[1]:
                        result[key] = model_input[1][key]
                return result
            else:
                result = {
                    "prediction": content,
                    "parse_error": "Failed to parse JSON",
                    "action": model_input[1].get("action", ""),
                    "image_path": model_input[1].get("image_path", ""),
                }
                for key in ["scenario_id", "risk_type", "mechanism", "hazard"]:
                    if key in model_input[1]:
                        result[key] = model_input[1][key]
                return result
        else:
            if self.task_name in json_prompts:
                parsed_json = parse_json_prediction(content)
                if parsed_json is not None:
                    return {
                        "prediction": parsed_json,
                        "raw_prediction": content,
                        "model_input": model_input[0],
                        **model_input[1],
                    }
                else:
                    return {
                        "prediction": content,
                        "parse_error": "Failed to parse JSON",
                        "model_input": model_input[0],
                        **model_input[1],
                    }
            else:
                return {
                    "prediction": content,
                    "model_input": model_input[0],
                    **model_input[1],
                }
    
    def get_generate_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for LLM generate call (e.g., image=True for VLM tasks)."""
        if self.task_name == "vqa":
            return {"image": True, "image_data": None}  # image_data will be set from model_input
        elif self.task_name == "guardrail":
            return {"image": True, "image_data": None}
        elif self.task_name in ["graph_to_image", "text_to_image"]:
            return {"image": True}
        else:
            return {}
    
    @abstractmethod
    def gather_results(self, results_file: str, scenarios_file: str, output_file: str, **kwargs) -> Dict[str, Any]:
        """
        Gather and merge results with original scenarios.
        
        Each task must implement its own gathering logic as the structure
        and requirements vary significantly between tasks.
        
        Args:
            results_file: Path to task results JSON file
            scenarios_file: Path to original scenarios JSON file
            output_file: Path to output merged file
            **kwargs: Additional task-specific arguments
        
        Returns:
            Dictionary with gathered results and metadata
        """
        pass


# Import all tasks to register them
try:
    from src.dataset_generation.tasks import (
        TaxonomyToScenarioTask,
        ScenarioToGraphTask,
        GraphToTextTask,
        TextToImageTask,
        SceneNormalizationTask,
        SceneAugmentationTask,
        HazardRemovalTask,
        HazardAugmentationTask,
        ActionAugmentationTask,
        EMBGuardTrainDataConstructionTask,
    )
    
    TASK_REGISTRY = {
        "taxonomy_to_scenario": TaxonomyToScenarioTask,
        "scenario_to_graph": ScenarioToGraphTask,
        "graph_to_text": GraphToTextTask,
        "text_to_image": TextToImageTask,
        "scene_normalization": SceneNormalizationTask,
        "scene_augmentation": SceneAugmentationTask,
        "hazard_removal": HazardRemovalTask,
        "hazard_augmentation": HazardAugmentationTask,
        "action_augmentation": ActionAugmentationTask,
        "embguard_train_data_construction": EMBGuardTrainDataConstructionTask,
    }
except ImportError:
    # Fallback if tasks are not yet created
    TASK_REGISTRY = {}


def get_task(task_name: str, args):
    """Factory function to get task instance by name."""
    task_class = TASK_REGISTRY.get(task_name)
    if task_class is None:
        raise ValueError(f"Unknown task: {task_name}. Available tasks: {list(TASK_REGISTRY.keys())}")
    return task_class(args)
