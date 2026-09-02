"""
Text to Image Task
Converts text descriptions into images.
"""
from typing import List, Dict, Any

from src.dataset_generation.task import BaseTask
from src.dataset_generation.prompters import GPTPrompter


class TextToImageTask(BaseTask):
    """Task: text → image"""
    
    def get_task_name(self) -> str:
        return "text_to_image"
    
    def load_data(self) -> List[Dict[str, Any]]:
        from src.dataset_generation.utils.data_loading import load_text_data
        import os
        
        # Auto-detect input file if not provided
        if not self.args.data_path:
            # Try to find texts_generated.json in iterate_output_dir
            if self.args.iterate_name:
                iterate_base = self.args.iterate_name.split('/')[0]
                iterate_output_dir = os.path.join(self.args.save_dir, iterate_base)
                texts_generated_path = os.path.join(iterate_output_dir, "texts_generated.json")
                if os.path.exists(texts_generated_path):
                    self.args.data_path = texts_generated_path
        
        if not self.args.data_path:
            raise ValueError("data_path not provided and could not be auto-detected. Please provide --data_path or ensure texts_generated.json exists in iterate_output_dir.")
        
        return load_text_data(self.args.data_path)
    
    def create_prompter(self):
        return GPTPrompter(self.args.prompt_path, self.task_name)
    
    def gather_results(self, results_file: str, scenarios_file: str, output_file: str, **kwargs) -> Dict[str, Any]:
        """
        Gather text-to-image results.
        For text_to_image task, results are typically handled by batch processing,
        so this is a minimal implementation.
        """
        import json
        import os
        
        # Load scenarios
        with open(scenarios_file, 'r', encoding='utf-8') as f:
            scenarios_data = json.load(f)
        
        scenarios = scenarios_data.get('scenarios', scenarios_data) if isinstance(scenarios_data, dict) else scenarios_data
        
        # Load results if available
        results = []
        if os.path.exists(results_file):
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
        
        result = {
            "metadata": {
                "total_scenarios": len(scenarios),
                "results_count": len(results) if isinstance(results, list) else 0
            },
            "scenarios": scenarios
        }
        
        # Save results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return result

