"""
Scenario to Graph Task
Converts scenario descriptions into graph representations.
"""
from typing import List, Dict, Any
import json
import os
import re

from src.dataset_generation.task import BaseTask
from src.dataset_generation.prompters import GPTPrompter


class ScenarioToGraphTask(BaseTask):
    """Task: scenario → graph"""
    
    def get_task_name(self) -> str:
        return "scenario_to_graph"
    
    def load_data(self) -> List[Dict[str, Any]]:
        from src.dataset_generation.utils.data_loading import load_scenario_data
        # Priority: scenarios_file > data_path > auto-detect
        data_path = None
        
        # First, try scenarios_file if provided
        if hasattr(self.args, 'scenarios_file') and self.args.scenarios_file:
            data_path = self.args.scenarios_file
            if not os.path.isabs(data_path) and self.args.iterate_name:
                # If relative path, resolve relative to iterate directory
                iterate_base = self.args.iterate_name.split('/')[0]
                iterate_dir = os.path.join(self.args.save_dir, iterate_base)
                data_path = os.path.join(iterate_dir, data_path)
        
        # Second, try data_path if provided
        if not data_path:
            data_path = self.args.data_path
        
        # Third, auto-detect from iterate directory
        if not data_path and self.args.iterate_name:
            iterate_base = self.args.iterate_name.split('/')[0]
            iterate_dir = os.path.join(self.args.save_dir, iterate_base)
            # Try scenarios_generated.json first, then scenarios.json
            scenarios_file = os.path.join(iterate_dir, "scenarios_generated.json")
            if not os.path.exists(scenarios_file):
                scenarios_file = os.path.join(iterate_dir, "scenarios.json")
            if os.path.exists(scenarios_file):
                data_path = scenarios_file
        
        if not data_path:
            raise ValueError("scenarios_file or data_path is required for scenario_to_graph task")
        return load_scenario_data(data_path)
    
    def create_prompter(self):
        return GPTPrompter(self.args.prompt_path, self.task_name)
    
    def gather_results(self, results_file: str, scenarios_file: str, output_file: str, **kwargs) -> Dict[str, Any]:
        """
        Gather scenarios with graphs from scenario_to_graph results.
        Integrated from gather_graphs.py
        """
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
            if not isinstance(item, dict):
                continue
            scenario_id = item.get('id', '')
            prediction = item.get('prediction', '')
            
            if scenario_id and prediction:
                graph = self._parse_graph_from_prediction(prediction)
                if graph is not None:
                    graph_mapping[scenario_id] = graph
        
        # Update scenarios with graphs
        updated_count = 0
        for scenario in existing_scenarios:
            scenario_id = scenario.get('id')
            if scenario_id in graph_mapping:
                scenario['graph'] = graph_mapping[scenario_id]
                updated_count += 1
        
        # Update metadata
        total_scenarios = len(existing_scenarios)
        scenarios_with_graphs = len([s for s in existing_scenarios if s.get('graph')])
        
        result = {
            "metadata": {
                **existing_data.get('metadata', {}),
                "total_scenarios": total_scenarios,
                "scenarios_with_graphs": scenarios_with_graphs,
                "scenarios_without_graphs": total_scenarios - scenarios_with_graphs,
                "updated_count": updated_count
            },
            "scenarios": existing_scenarios
        }
        
        # Save results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return result
    
    def _parse_graph_from_prediction(self, prediction: Any) -> Any:
        """Parse graph from prediction string"""
        try:
            if isinstance(prediction, list) and len(prediction) > 0:
                pred_item = prediction[0]
                if isinstance(pred_item, dict) and 'graph' in pred_item:
                    return pred_item['graph']
                elif isinstance(pred_item, str):
                    try:
                        pred_json = json.loads(pred_item)
                        if 'graph' in pred_json:
                            return pred_json['graph']
                    except json.JSONDecodeError:
                        pass
            elif isinstance(prediction, str):
                try:
                    pred_json = json.loads(prediction)
                    if 'graph' in pred_json:
                        return pred_json['graph']
                except json.JSONDecodeError:
                    pass
            elif isinstance(prediction, dict) and 'graph' in prediction:
                return prediction['graph']
        except Exception:
            pass
        return None

