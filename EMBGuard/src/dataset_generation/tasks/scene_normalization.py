"""
Scene Normalization Task
Normalizes scene graphs to a standard format.
"""
from typing import List, Dict, Any

from src.dataset_generation.task import BaseTask
from src.dataset_generation.prompters import GraphPostprocessingPrompter


class SceneNormalizationTask(BaseTask):
    """Task: scene normalization"""
    
    def get_task_name(self) -> str:
        return "scene_normalization"
    
    def load_data(self) -> List[Dict[str, Any]]:
        from src.dataset_generation.utils.data_loading import load_graph_data
        # Use data_path if provided, otherwise try to find graphs.json in iterate directory
        data_path = self.args.data_path
        if not data_path and self.args.iterate_name:
            import os
            iterate_base = self.args.iterate_name.split('/')[0]
            iterate_dir = os.path.join(self.args.save_dir, iterate_base)
            graphs_file = os.path.join(iterate_dir, "graphs.json")
            if os.path.exists(graphs_file):
                data_path = graphs_file
        if not data_path:
            raise ValueError("data_path is required for scene_normalization task")
        return load_graph_data(data_path)
    
    def create_prompter(self):
        return GraphPostprocessingPrompter(self.args.prompt_path, self.task_name)
    
    def gather_results(self, results_file: str, scenarios_file: str, output_file: str, **kwargs) -> Dict[str, Any]:
        """
        Gather normalized graphs (similar to scenario_to_graph).
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

