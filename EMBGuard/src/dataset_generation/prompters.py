import logging
from typing import Generator, Optional, Union
import yaml
import json
import os
import sys
from colorama import Fore
import base64

# Add project root to Python path (needed before importing utils)
current_dir = os.path.dirname(os.path.abspath(__file__))
# current_dir is src/dataset_generation, so we need to go up 2 levels to get project root
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now we can import utils - use try/except for safety
try:
    from utils.path import get_project_path
except ImportError:
    # Fallback: define get_project_path locally if utils.path is not available
    from pathlib import Path
    def get_project_path():
        return Path(__file__).parents[2]  # Go up from src/dataset_generation/prompters.py to project root

from src.dataset_generation.utils.img_utils import compress_and_encode_image_to_base64, encode_image_to_base64

IGNORE_TOKEN_ID = -100


class Prompter:
    """
    Base prompter class for all prompters
    """
    
class GPTPrompter(Prompter):
    def __init__(self, prompt:str, prompt_key:str):
        with open(prompt, "r", encoding="UTF-8") as f:
            self.prompt = yaml.load(f, Loader=yaml.FullLoader)[prompt_key]
            print(self.prompt)
    
    def build_prompt(self, input_dict:dict):
        return self.prompt.format(**input_dict)


class RiskPipelinePrompter(GPTPrompter):
    """
    Prompter for risk pipeline that loads examples from shots.json dynamically.
    Extends GPTPrompter to add example generation from shots.json based on risk type.
    """
    def __init__(self, prompt: str, prompt_key: str, shots_path: Optional[str] = None, taxonomy_path: Optional[str] = None):
        super().__init__(prompt, prompt_key)
        
        # Load shots.json
        if shots_path is None:
            # Try to get from config.yaml
            try:
                from utils.config import get_config
                config = get_config()
                dataset_gen_config = config.get("dataset_generation", {})
                shots_file = dataset_gen_config.get("shots_file", "shots_w_mechanism.json")
                source_dir = dataset_gen_config.get("source_dir", "src/dataset_generation/resources/source")
                
                # If shots_file is absolute, use it directly
                if os.path.isabs(shots_file):
                    shots_path = shots_file
                else:
                    # Combine with source_dir
                    if os.path.isabs(source_dir):
                        shots_path = os.path.join(source_dir, shots_file)
                    else:
                        # Get project root
                        current_dir = os.path.dirname(os.path.abspath(__file__))
                        project_root = os.path.dirname(os.path.dirname(current_dir))
                        shots_path = os.path.join(project_root, source_dir, shots_file)
            except ImportError:
                # Fallback: use relative path from prompt file
                prompt_dir = os.path.dirname(prompt)
                shots_path = os.path.join(prompt_dir, "..", "source", "shots_w_mechanism.json")
                shots_path = os.path.normpath(shots_path)
        
        self.shots = None
        if os.path.exists(shots_path):
            with open(shots_path, "r", encoding="UTF-8") as f:
                self.shots = json.load(f)
        else:
            print(f"Warning: shots.json not found at {shots_path}")
        
        # Load taxonomy.json
        if taxonomy_path is None:
            # Default path: res/source/taxonomy.json relative to prompt file
            prompt_dir = os.path.dirname(prompt)
            taxonomy_path = os.path.join(prompt_dir, "..", "source", "taxonomy.json")
            taxonomy_path = os.path.normpath(taxonomy_path)
        
        self.taxonomy = None
        if os.path.exists(taxonomy_path):
            with open(taxonomy_path, "r", encoding="UTF-8") as f:
                self.taxonomy = json.load(f)
        else:
            print(f"Warning: taxonomy.json not found at {taxonomy_path}")

        # Load tips_for_mechanism.json
        tips_path = os.path.join(os.path.dirname(taxonomy_path), "tips_for_mechanism.json")
        tips_path = os.path.normpath(tips_path)
        self.tips_for_mechanism = None
        if os.path.exists(tips_path):
            with open(tips_path, "r", encoding="UTF-8") as f:
                self.tips_for_mechanism = json.load(f)
        else:
            print(f"Warning: tips_for_mechanism.json not found at {tips_path}")
    
    def _generate_example_input(self, risk_type: str) -> str:
        """Generate example input string from taxonomy.json based on risk type"""
        if not self.taxonomy:
            return ""
        
        if risk_type not in self.taxonomy:
            return ""

        taxonomy_data = self.taxonomy[risk_type]
        description = taxonomy_data.get("description", "")
        
        # Get all mechanisms for example
        mechanisms = taxonomy_data.get("mechanism", [])
        # Format mechanisms as JSON string
        if mechanisms:
            mechanism_str = json.dumps(mechanisms, ensure_ascii=False)
        else:
            mechanism_str = "[]"
        
        # Format as example input
        example_input = f'- "type": {risk_type}\n  - "description": {description}\n  - "mechanism": {mechanism_str}'
        
        return example_input
    
    def _generate_examples_from_shots(self, risk_type: str, mechanism_name: Optional[str] = None) -> str:
        """Generate example JSON string from shots.json based on risk type"""
        if not self.shots:
            return ""
        
        if risk_type not in self.shots:
            return ""
        
        risk_examples = self.shots[risk_type]
        examples = []
        if isinstance(risk_examples, dict):
            if mechanism_name and mechanism_name in risk_examples:
                examples = risk_examples[mechanism_name]
            elif mechanism_name is None:
                # If mechanism not specified, gather all examples from every mechanism
                for mech_examples in risk_examples.values():
                    examples.extend(mech_examples)
        elif isinstance(risk_examples, list):
            examples = risk_examples
        
        if not examples:
            return ""
        
        # Format examples as JSON string matching the prompt format
        # Use {{{{ to get {{ in final output (needed for Python's .format() method)
        scenarios_json = "  {{\n    \"scenarios\": [\n"
        scenario_items = []
        for example in examples:
            content = example.get("hazard", example.get("hazard", ""))
            action = example.get("action", "")
            # Escape quotes in content and action
            content_escaped = content.replace('"', '\\"')
            action_escaped = action.replace('"', '\\"')
            # Use {{{{ to get {{ in final output (needed for Python's .format() method)
            scenario_items.append(
                f"      {{{{}}\n        \"hazard\": \"{content_escaped}\",\n        \"action\": \"{action_escaped}\"\n      {{{{}}}}"
            )
        scenarios_json += ",\n".join(scenario_items) + "\n    ]\n  }}"
        
        return scenarios_json
    
    def build_prompt(self, input_dict: dict):
        # Generate example input and examples dynamically from taxonomy.json and shots.json if type is provided
        if "type" in input_dict:
            risk_type = input_dict["type"]
            
            # Generate example input from taxonomy.json
            if self.taxonomy:
                example_input = self._generate_example_input(risk_type)
                input_dict["example_input"] = example_input
            
            # Generate examples from shots.json
            if self.shots:
                mechanism_name = None
                mechanism_data = input_dict.get("mechanism")
                if isinstance(mechanism_data, list) and mechanism_data:
                    first_entry = mechanism_data[0]
                    if isinstance(first_entry, dict):
                        mechanism_name = first_entry.get("name")
                elif isinstance(mechanism_data, dict):
                    mechanism_name = mechanism_data.get("name")

                examples = self._generate_examples_from_shots(risk_type, mechanism_name)
                input_dict["examples"] = examples

                # Attach mechanism tip if available
                tip_text = ""
                if self.tips_for_mechanism and mechanism_name:
                    tip_item = self.tips_for_mechanism.get(mechanism_name)
                    if isinstance(tip_item, dict):
                        tip_text = tip_item.get("tip", "") or ""
                input_dict["tip"] = tip_text
            
            # Format mechanism for prompt (convert to JSON string if dict/list)
            # mechanism is already included in input_dict from load_taxonomy_data
            if "mechanism" in input_dict:
                if isinstance(input_dict["mechanism"], (dict, list)):
                    input_dict["mechanism"] = json.dumps(input_dict["mechanism"], ensure_ascii=False)

        # Ensure optional placeholders exist
        input_dict.setdefault("tip", "")
        
        return self.prompt.format(**input_dict)


class GraphPostprocessingPrompter(GPTPrompter):
    """
    Prompter for graph post-processing tasks.
    """
    def __init__(self, prompt: str, prompt_key: str):
        super().__init__(prompt, prompt_key)
        self.prompt_key = prompt_key

    def build_prompt(self, input_dict: dict):
        return self.prompt.format(**input_dict)


class VLMGPTPrompter(GPTPrompter):
    def __init__(self, prompt:str, prompt_key:str):
        super().__init__(prompt, prompt_key)
        
    def build_prompt(self, input_dict:dict):
        input_dict["image"] = compress_and_encode_image_to_base64(input_dict["image_path"])
        return self.prompt.format(**input_dict) 


