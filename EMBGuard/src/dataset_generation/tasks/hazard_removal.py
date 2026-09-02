"""
Hazard Removal Task
Removes hazards from scenes to create safe scenarios.
"""
from typing import List, Dict, Any

from src.dataset_generation.task import BaseTask
from src.dataset_generation.prompters import GraphPostprocessingPrompter


class HazardRemovalTask(BaseTask):
    """Task: hazard removal"""
    
    def get_task_name(self) -> str:
        return "hazard_removal"
    
    def load_data(self) -> List[Dict[str, Any]]:
        from src.dataset_generation.utils.data_loading import load_graph_data
        import os
        # Priority: data_path > auto-detect
        data_path = self.args.data_path
        
        # If data_path is provided but relative, resolve relative to iterate directory
        if data_path and not os.path.isabs(data_path) and self.args.iterate_name:
            iterate_base = self.args.iterate_name.split('/')[0]
            iterate_dir = os.path.join(self.args.save_dir, iterate_base)
            # Check if file exists in iterate directory
            potential_path = os.path.join(iterate_dir, data_path)
            if os.path.exists(potential_path):
                data_path = potential_path
        
        # Auto-detect from iterate directory if not provided
        if not data_path and self.args.iterate_name:
            iterate_base = self.args.iterate_name.split('/')[0]
            iterate_dir = os.path.join(self.args.save_dir, iterate_base)
            # Try graphs_scene_augmented.json first, then graphs_normalized.json, then graphs.json
            graphs_file = os.path.join(iterate_dir, "graphs_scene_augmented.json")
            if not os.path.exists(graphs_file):
                graphs_file = os.path.join(iterate_dir, "graphs_normalized.json")
            if not os.path.exists(graphs_file):
                graphs_file = os.path.join(iterate_dir, "graphs.json")
            if os.path.exists(graphs_file):
                data_path = graphs_file
        
        if not data_path:
            raise ValueError("data_path is required for hazard_removal task")
        
        # Store data_path for later use in gather_results
        self._input_data_path = data_path
        return load_graph_data(data_path)
    
    def create_prompter(self):
        return GraphPostprocessingPrompter(self.args.prompt_path, self.task_name)
    
    def gather_results(self, results_file: str, scenarios_file: str, output_file: str, **kwargs) -> Dict[str, Any]:
        """
        Gather hazard removed graphs (similar to scenario_to_graph).
        """
        import json
        import os
        
        # Load existing scenarios
        with open(scenarios_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        existing_scenarios = existing_data.get('scenarios', [])
        
        # Load graph results
        if not os.path.exists(results_file):
            return existing_data
        
        with open(results_file, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        if not isinstance(graph_data, list):
            return existing_data
        
        # Create ID-to-graph mapping
        graph_mapping = {}
        for item in graph_data:
            if isinstance(item, dict):
                scenario_id = item.get('id', '')
                prediction = item.get('prediction', '')
                if scenario_id and prediction:
                    if isinstance(prediction, list) and len(prediction) > 0:
                        pred_data = prediction[0]
                        if isinstance(pred_data, dict) and 'graph' in pred_data:
                            graph_mapping[scenario_id] = pred_data['graph']
                    elif isinstance(prediction, dict) and 'graph' in prediction:
                        graph_mapping[scenario_id] = prediction['graph']
        
        # Update scenarios with graphs
        for scenario in existing_scenarios:
            scenario_id = scenario.get('id')
            if scenario_id in graph_mapping:
                scenario['graph'] = graph_mapping[scenario_id]
        
        result = {
            "metadata": {
                **existing_data.get('metadata', {}),
                "total_scenarios": len(existing_scenarios),
                "scenarios_with_graphs": len([s for s in existing_scenarios if s.get('graph')])
            },
            "scenarios": existing_scenarios
        }
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return result

