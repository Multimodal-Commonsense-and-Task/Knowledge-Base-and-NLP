"""
Action Augmentation Task
Augments scenarios by replacing actions with safer alternatives.
"""
from typing import List, Dict, Any
import json
import os

from src.dataset_generation.task import BaseTask
from src.dataset_generation.prompters import GraphPostprocessingPrompter


class ActionAugmentationTask(BaseTask):
    """Task: action augmentation"""
    
    def get_task_name(self) -> str:
        return "action_augmentation"
    
    def load_data(self) -> List[Dict[str, Any]]:
        from src.dataset_generation.utils.data_loading import load_graph_data
        import os
        
        # Auto-detect input file if not provided
        if not self.args.data_path:
            # Try to find texts_with_images_*.json files in iterate_output_dir
            if self.args.iterate_name:
                iterate_base = self.args.iterate_name.split('/')[0]
                iterate_output_dir = os.path.join(self.args.save_dir, iterate_base)
                
                # Look for texts_with_images_*.json files
                import glob
                pattern = os.path.join(iterate_output_dir, "texts_with_images_*.json")
                matching_files = glob.glob(pattern)
                
                if matching_files:
                    # Use the first matching file (or could prioritize by merge_source)
                    self.args.data_path = matching_files[0]
        
        if not self.args.data_path:
            raise ValueError("data_path not provided and could not be auto-detected. Please provide --data_path or ensure texts_with_images_*.json exists in iterate_output_dir.")
        
        return load_graph_data(self.args.data_path)
    
    def create_prompter(self):
        return GraphPostprocessingPrompter(self.args.prompt_path, self.task_name)
    
    def gather_results(self, results_file: str, scenarios_file: str, output_file: str, **kwargs) -> Dict[str, Any]:
        """
        Gather action-augmented scenarios.
        Integrated from gather_actions.py
        """
        include_original = kwargs.get('include_original', True)
        
        # Load original scenarios
        with open(scenarios_file, 'r', encoding='utf-8') as f:
            scenarios_data = json.load(f)
        original_scenarios = scenarios_data.get('scenarios', scenarios_data) if isinstance(scenarios_data, dict) else scenarios_data
        
        # Load action augmentation results
        action_map = {}
        if os.path.exists(results_file):
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'id' in item:
                        scenario_id = item['id']
                        if 'prediction' in item:
                            predictions = item['prediction']
                            # Handle both dict and list formats
                            if isinstance(predictions, dict) and 'action' in predictions:
                                action_map[scenario_id] = predictions['action']
                            elif isinstance(predictions, list) and len(predictions) > 0:
                                pred = predictions[0]
                                if isinstance(pred, dict) and 'action' in pred:
                                    action_map[scenario_id] = pred['action']
                        elif 'action' in item:
                            action_map[scenario_id] = item['action']
        
        # Create augmented scenarios
        scenario_dict = {s['id']: s for s in original_scenarios}
        augmented_scenarios = []
        augmentation_counter = {}
        
        for scenario_id, new_action in action_map.items():
            if scenario_id in scenario_dict:
                original = scenario_dict[scenario_id]
                if scenario_id not in augmentation_counter:
                    augmentation_counter[scenario_id] = 0
                augmentation_counter[scenario_id] += 1
                
                aug_index = augmentation_counter[scenario_id]
                augmented_id = f"{scenario_id}_aug_{aug_index:03d}"
                
                augmented = {
                    "id": augmented_id,
                    "risk_type": original['risk_type'],
                    "mechanism": original['mechanism'],
                    "hazard": original['hazard'],
                    "action": new_action,
                    "source": "action_augmented",
                    "original_id": scenario_id
                }
                
                if 'graph' in original:
                    augmented['graph'] = original['graph']
                
                for key in original:
                    if key not in augmented and key not in ['source', 'action']:
                        augmented[key] = original[key]
                
                augmented_scenarios.append(augmented)
        
        # Combine scenarios
        all_scenarios = []
        if include_original:
            all_scenarios.extend(original_scenarios)
        all_scenarios.extend(augmented_scenarios)
        
        result = {
            "metadata": {
                "total_scenarios": len(all_scenarios),
                "original_count": len([s for s in all_scenarios if s.get('source') != 'action_augmented']),
                "augmented_count": len([s for s in all_scenarios if s.get('source') == 'action_augmented']),
                "scenarios_with_augmentations": len(augmentation_counter),
                "include_original": include_original
            },
            "scenarios": all_scenarios
        }
        
        # Save results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return result

