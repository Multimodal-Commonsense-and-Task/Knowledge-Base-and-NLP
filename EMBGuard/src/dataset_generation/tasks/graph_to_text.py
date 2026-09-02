"""
Graph to Text Task
Converts graph representations into text descriptions.
"""
from typing import List, Dict, Any
import json
import os
from collections import Counter

from src.dataset_generation.task import BaseTask
from src.dataset_generation.prompters import GPTPrompter


class GraphToTextTask(BaseTask):
    """Task: graph → text"""
    
    def get_task_name(self) -> str:
        return "graph_to_text"
    
    def load_data(self) -> List[Dict[str, Any]]:
        from src.dataset_generation.utils.data_loading import load_graph_data
        import os
        
        if not self.args.data_path:
            raise ValueError("data_path is required. Please provide --data_path in the script.")
        
        return load_graph_data(self.args.data_path)
    
    def create_prompter(self):
        return GPTPrompter(self.args.prompt_path, self.task_name)
    
    def gather_results(self, results_file: str, scenarios_file: str, output_file: str, **kwargs) -> Dict[str, Any]:
        """
        Gather scenarios with text descriptions from graph_to_text results.
        Integrated from gather_texts.py
        """
        # Load scenarios
        with open(scenarios_file, 'r', encoding='utf-8') as f:
            scenarios_data = json.load(f)
        scenarios = scenarios_data.get('scenarios', scenarios_data) if isinstance(scenarios_data, dict) else scenarios_data
        
        # Load text results
        text_lookup = {}
        if os.path.exists(results_file):
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        scenario_id = item.get('id', '')
                        merge_source = item.get('merge_source', item.get('source', 'unknown'))
                        unique_key = f"{scenario_id}||{merge_source}"
                        
                        pred = item.get('prediction', '')
                        if isinstance(pred, str):
                            text_lookup[unique_key] = pred.strip()
                        elif isinstance(pred, list) and len(pred) > 0:
                            if isinstance(pred[0], str):
                                text_lookup[unique_key] = pred[0].strip()
                            elif isinstance(pred[0], dict):
                                text_lookup[unique_key] = pred[0].get('text', pred[0].get('description', str(pred[0]))).strip()
        
        # Add situation descriptions
        scenarios_with_texts = []
        scenarios_without_texts = []
        
        for scenario in scenarios:
            scenario_id = scenario.get('id', '')
            merge_source = scenario.get('merge_source', scenario.get('source', 'unknown'))
            unique_key = f"{scenario_id}||{merge_source}"
            
            if unique_key in text_lookup:
                scenario['situation'] = text_lookup[unique_key]
                scenarios_with_texts.append(scenario)
            else:
                scenario['situation'] = None
                scenarios_without_texts.append(scenario)
        
        risk_type_counts = Counter(s['risk_type'] for s in scenarios_with_texts if 'risk_type' in s)
        
        result = {
            'metadata': {
                'total_scenarios': len(scenarios),
                'scenarios_with_texts': len(scenarios_with_texts),
                'scenarios_without_texts': len(scenarios_without_texts),
                'success_rate': len(scenarios_with_texts) / len(scenarios) if scenarios else 0,
                'risk_type_counts': dict(risk_type_counts)
            },
            'scenarios': scenarios_with_texts + scenarios_without_texts
        }
        
        # Save results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return result

