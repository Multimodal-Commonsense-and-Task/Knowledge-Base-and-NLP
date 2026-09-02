"""
Scene Augmentation Task
Augments scenes by adding new graph elements.
"""
from typing import List, Dict, Any
import json
import os

from src.dataset_generation.task import BaseTask
from src.dataset_generation.prompters import GraphPostprocessingPrompter


class SceneAugmentationTask(BaseTask):
    """Task: scene augmentation"""
    
    def get_task_name(self) -> str:
        return "scene_augmentation"
    
    def load_data(self) -> List[Dict[str, Any]]:
        from src.dataset_generation.utils.data_loading import load_graph_data
        # Use data_path if provided, otherwise try to find graphs_normalized.json in iterate directory
        data_path = self.args.data_path
        if not data_path and self.args.iterate_name:
            iterate_base = self.args.iterate_name.split('/')[0]
            iterate_dir = os.path.join(self.args.save_dir, iterate_base)
            graphs_file = os.path.join(iterate_dir, "graphs_normalized.json")
            if os.path.exists(graphs_file):
                data_path = graphs_file
        if not data_path:
            raise ValueError("data_path is required for scene_augmentation task")
        return load_graph_data(data_path)
    
    def create_prompter(self):
        return GraphPostprocessingPrompter(self.args.prompt_path, self.task_name)
    
    def gather_results(self, results_file: str, scenarios_file: str, output_file: str, **kwargs) -> Dict[str, Any]:
        """
        Gather scene augmented graphs (handles candidates merging).
        """
        # Load existing scenarios
        with open(scenarios_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        existing_scenarios = existing_data.get('scenarios', [])
        
        # Load augmentation results
        if not os.path.exists(results_file):
            return existing_data
        
        with open(results_file, 'r', encoding='utf-8') as f:
            aug_data = json.load(f)
        
        if not isinstance(aug_data, list):
            return existing_data
        
        # Parse candidates from predictions
        candidates_map = {}
        for item in aug_data:
            if isinstance(item, dict):
                scenario_id = item.get('id', '')
                prediction = item.get('prediction', '')
                candidates = self._parse_candidates_from_prediction(prediction)
                if scenario_id and candidates:
                    candidates_map[scenario_id] = candidates
        
        # Merge candidates into existing graphs and update ids
        for scenario in existing_scenarios:
            scenario_id = scenario.get('id')
            original_id = scenario_id
            
            # Append _scene_aug to id and store original id
            if scenario_id and not scenario_id.endswith('_scene_aug'):
                scenario['id'] = f"{scenario_id}_scene_aug"
                scenario['org_id'] = original_id
            
            # Use original_id for candidates_map lookup
            if original_id in candidates_map:
                existing_graph = scenario.get('graph', [])
                candidates = candidates_map[original_id]
                scenario['graph'] = self._merge_candidates_into_graph(existing_graph, candidates)
                scenario['scene_augmentation_candidates'] = candidates
        
        result = {
            "metadata": {
                **existing_data.get('metadata', {}),
                "total_scenarios": len(existing_scenarios),
                "scenarios_with_augmentations": len(candidates_map)
            },
            "scenarios": existing_scenarios
        }
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return result
    
    def _parse_candidates_from_prediction(self, prediction: Any) -> Any:
        """Parse scene_augmentation candidates from prediction."""
        try:
            if isinstance(prediction, list) and prediction:
                if isinstance(prediction[0], dict):
                    prediction = prediction[0]
            
            if isinstance(prediction, str):
                try:
                    prediction = json.loads(prediction)
                except json.JSONDecodeError:
                    return None
            
            if isinstance(prediction, dict):
                candidates = prediction.get("candidates")
                if isinstance(candidates, list):
                    parsed = []
                    for triple in candidates:
                        if isinstance(triple, list) and len(triple) >= 3:
                            parsed.append([triple[0], triple[1], triple[2]])
                    return parsed if parsed else None
        except Exception:
            pass
        return None
    
    def _merge_candidates_into_graph(self, graph: Any, candidates: List) -> List:
        """Append candidate triplets, removing exact duplicates."""
        base = graph or []
        seen = set()
        merged = []
        for triple in base:
            if isinstance(triple, list) and len(triple) >= 3:
                t = tuple(triple[:3])
                if t not in seen:
                    merged.append(list(t))
                    seen.add(t)
        for triple in candidates:
            if isinstance(triple, list) and len(triple) >= 3:
                t = tuple(triple[:3])
                if t not in seen:
                    merged.append(list(t))
                    seen.add(t)
        return merged

